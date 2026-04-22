#!/usr/bin/env python3
"""
Pure-ONNX inference backend for Qwen3-ASR (INT4 MatMulNBits).

Weights come from:
  - andrewleech/qwen3-asr-0.6b-onnx   (~2.0 GB disk,  ~1.0 GB RSS at runtime)
  - andrewleech/qwen3-asr-1.7b-onnx   (~4.0 GB disk,  ~1.3 GB RSS at runtime)

Why this backend:
  - INT4 MatMulNBits on the decoder keeps weight memory ~4x smaller than BF16.
  - The `decoder_weights.int4.data` file is memory-mapped by ONNX Runtime, so
    only hot pages are resident — steady-state RSS is dominated by the embedding
    table (0.5–0.6 GB) and activations, not by weight storage.
  - No PyTorch at runtime. Only onnxruntime + numpy + tokenizers.

Pipeline:
  1. Audio (float32, 16 kHz mono) -> log-mel [1, 128, T] (Whisper-compatible)
  2. encoder.int4.onnx  : mel -> audio_features [1, N, H]
  3. Build prompt IDs: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
     <|audio_start|><|audio_pad|> x N <|audio_end|><|im_end|>\n<|im_start|>assistant\n
     (optionally followed by "language {Lang} <asr_text>")
  4. decoder_init.int4.onnx : input_ids + position_ids + audio_features + audio_offset
                              -> logits, kv_cache  (audio scatter happens inside graph)
  5. Greedy decode loop with decoder_step.int4.onnx until EOS:
        embed_tokens[next_id] -> input_embeds -> decoder_step -> argmax
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Tokenizer-dependent constants (stable across 0.6B and 1.7B — verified by the
# upstream export tool in andrewleech/qwen3-asr-onnx/src/prompt.py).
# ----------------------------------------------------------------------------

ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ASR_TEXT_TOKEN_ID = 151704

SYSTEM_TOKEN_ID = 9125
USER_TOKEN_ID = 882
ASSISTANT_TOKEN_ID = 77091
NEWLINE_TOKEN_ID = 198

EOS_TOKEN_IDS = {ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID}

# Mel spectrogram (identical to Whisper — fixed for all Qwen3-ASR sizes)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CONV_WINDOW = 100
TOKENS_PER_WINDOW = 13


# ----------------------------------------------------------------------------
# Whisper-compatible log-mel spectrogram in pure NumPy.
# ----------------------------------------------------------------------------

def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    """Slaney-style HTK mel scale (used by Whisper / librosa default)."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    # np.where evaluates both branches; clamp below to avoid log(0) warnings.
    safe = np.maximum(hz, min_log_hz)
    mel = np.where(
        hz >= min_log_hz,
        min_log_mel + np.log(safe / min_log_hz) / logstep,
        (hz - f_min) / f_sp,
    )
    return mel


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        f_min + f_sp * mel,
    )
    return hz


def _build_mel_filterbank() -> np.ndarray:
    """
    Build the Whisper-compatible 128-bin Slaney mel filterbank as
    a [n_mels, n_fft // 2 + 1] matrix, normalized per-bin by bandwidth.
    """
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_freqs)

    # Edges in mel space: n_mels + 2 evenly-spaced points between 0 and Nyquist.
    min_mel = _hz_to_mel_slaney(np.array([0.0]))[0]
    max_mel = _hz_to_mel_slaney(np.array([SAMPLE_RATE / 2.0]))[0]
    mel_points = np.linspace(min_mel, max_mel, N_MELS + 2)
    hz_points = _mel_to_hz_slaney(mel_points)

    filters = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]

    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        filters[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization — scale each filter by 2 / (bandwidth in Hz).
    enorm = 2.0 / (hz_points[2 : N_MELS + 2] - hz_points[:N_MELS])
    filters *= enorm[:, None]
    return filters


_MEL_FILTERS: Optional[np.ndarray] = None
_HANN_WINDOW: Optional[np.ndarray] = None


def _get_mel_filters() -> np.ndarray:
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = _build_mel_filterbank()
    return _MEL_FILTERS


def _get_hann_window() -> np.ndarray:
    global _HANN_WINDOW
    if _HANN_WINDOW is None:
        # Periodic Hann (matches torch.hann_window / librosa default).
        _HANN_WINDOW = (
            0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
        ).astype(np.float32)
    return _HANN_WINDOW


def _stft(wav: np.ndarray) -> np.ndarray:
    """
    Center-padded STFT with reflect padding (matches librosa.stft / torch.stft
    center=True, pad_mode='reflect'). Returns complex magnitudes^2 in a
    [n_freqs, n_frames] float32 array.
    """
    pad = N_FFT // 2
    # reflect-pad without duplicating the endpoint (numpy default behaviour).
    padded = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH

    # Frame -> window -> rFFT. Using np.lib.stride_tricks avoids a Python loop.
    from numpy.lib.stride_tricks import as_strided

    stride = padded.strides[0]
    frames = as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(stride * HOP_LENGTH, stride),
    )
    windowed = frames * _get_hann_window()
    spec = np.fft.rfft(windowed, n=N_FFT, axis=1).T
    # Power spectrum.
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)


def compute_log_mel(wav: np.ndarray) -> np.ndarray:
    """
    Produce a [1, 128, T] log-mel spectrogram tensor in Whisper's normalized
    scale (clipped to the top 8 dB dynamic range, shifted to ~[0, 1]).
    """
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32, copy=False)
    power = _stft(wav)
    mel = _get_mel_filters() @ power
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel[np.newaxis, :, :].astype(np.float32)


# ----------------------------------------------------------------------------
# Encoder output length — branch-free formula from the upstream export wrapper.
# Must match exactly, otherwise the scatter inside decoder_init will mis-align.
# ----------------------------------------------------------------------------

def _conv_out_len(t: int) -> int:
    return (t + 1) // 2


def get_encoder_output_length(mel_frames: int) -> int:
    leave = mel_frames % CONV_WINDOW
    t = _conv_out_len(_conv_out_len(_conv_out_len(leave)))
    return t + (mel_frames // CONV_WINDOW) * TOKENS_PER_WINDOW


# ----------------------------------------------------------------------------
# Model assets — download & locate files on disk.
# ----------------------------------------------------------------------------

ONNX_REPO_MAP = {
    "0.6B": "andrewleech/qwen3-asr-0.6b-onnx",
    "1.7B": "andrewleech/qwen3-asr-1.7b-onnx",
}


REQUIRED_FILES = (
    "encoder.int4.onnx",
    "decoder_init.int4.onnx",
    "decoder_init.int4.onnx.data",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
    "embed_tokens.bin",
    "tokenizer.json",
    "config.json",
)


def download_onnx_model(size: str) -> Path:
    """
    Download the INT4 ONNX bundle for 0.6B or 1.7B and return the cache dir.
    Uses huggingface_hub snapshot_download with resume support.
    """
    if size not in ONNX_REPO_MAP:
        raise ValueError(f"Unknown size {size!r}, expected one of {list(ONNX_REPO_MAP)}")

    from huggingface_hub import snapshot_download

    # allow_patterns keeps the download surgical — we don't need the FP32
    # decoder weights (~2 GB for 0.6B, ~6 GB for 1.7B) when running INT4.
    allow_patterns = [
        "encoder.int4.onnx",
        "decoder_init.int4.onnx",
        "decoder_init.int4.onnx.data",
        "decoder_step.int4.onnx",
        "decoder_weights.int4.data",
        "embed_tokens.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "preprocessor_config.json",
    ]

    cache_dir = snapshot_download(
        repo_id=ONNX_REPO_MAP[size],
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    return Path(cache_dir)


# ----------------------------------------------------------------------------
# Pipeline class — owns ORT sessions, embedding matrix, tokenizer, and config.
# ----------------------------------------------------------------------------

@dataclass
class AsrResult:
    text: str
    language: str


class Qwen3AsrOnnx:
    """
    End-to-end Qwen3-ASR inference using INT4 ONNX weights.

    Memory footprint (1.7B, after warmup, RSS on macOS arm64):
        ~1.2–1.5 GB steady-state vs ~5–6 GB for PyTorch BF16.

    The heavy files are memory-mapped by ORT (decoder_weights.int4.data) so
    the kernel only pages in what's being actively used.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        num_threads: int = 0,
        provider: str = "cpu",
    ):
        import onnxruntime as ort

        self._ort = ort
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self._validate_files(model_dir)

        # --- Load config ---
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.hidden_size: int = int(cfg["decoder"]["hidden_size"])
        self.vocab_size: int = int(cfg["decoder"]["vocab_size"])
        self.audio_hidden_size: int = int(
            cfg["encoder"].get("output_dim", self.hidden_size)
        )

        # --- Tokenizer (native Rust, ~80 MB RAM) ---
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        # --- Embedding table (FP16 on disk; cast to FP32 once at startup) ---
        # [vocab_size, hidden_size]. 1.7B => 151936 * 2048 * 4 bytes = ~1.24 GB
        # in FP32, or we can keep FP16 and cast on demand to save ~0.6 GB.
        embed_bytes = np.fromfile(
            model_dir / "embed_tokens.bin", dtype=np.float16
        )
        embed_fp16 = embed_bytes.reshape(self.vocab_size, self.hidden_size)
        # Keep FP16 to save RAM. Single-token lookups are cheap to cast.
        self._embed_tokens_fp16 = embed_fp16

        # --- Session options ---
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
            so.inter_op_num_threads = 1

        providers: list[str | Tuple[str, dict]] = []
        if provider == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # --- ORT sessions ---
        self.encoder_sess = ort.InferenceSession(
            str(model_dir / "encoder.int4.onnx"),
            sess_options=so,
            providers=providers,
        )
        self.decoder_init_sess = ort.InferenceSession(
            str(model_dir / "decoder_init.int4.onnx"),
            sess_options=so,
            providers=providers,
        )
        self.decoder_step_sess = ort.InferenceSession(
            str(model_dir / "decoder_step.int4.onnx"),
            sess_options=so,
            providers=providers,
        )

        # Cache the encoded "language {Name}<asr_text>" token prefixes — the
        # same ones qwen-asr generates via its chat template for forced lang.
        self._lang_prefix_cache: dict[str, list[int]] = {}

    def _validate_files(self, model_dir: Path) -> None:
        missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing ONNX files in {model_dir}: {missing}\n"
                "Did you run `python download_model.py` and pick an ONNX option?"
            )

    # -------------------- public API -----------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 100,
    ) -> AsrResult:
        """
        Transcribe a mono 16 kHz float32 audio array.

        Args:
            audio: 1-D numpy array of float32 samples in [-1, 1].
            language: Optional full name ("English", "Hindi", ...). When set,
                the prompt forces the model to emit transcription-only output.
            max_new_tokens: Hard cap on generated tokens (the prompt prefix
                does not count against this).
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        # 1. Mel
        mel = compute_log_mel(audio)  # [1, 128, T]

        # 2. Encoder -> audio features [1, N, H_audio]
        audio_features = self.encoder_sess.run(
            None, {"mel": mel}
        )[0]
        n_audio_tokens = audio_features.shape[1]

        # Sanity: N_audio_tokens must equal get_encoder_output_length(mel_frames)
        # The decoder_init scatter uses audio_offset and audio_features.shape[1]
        # directly, so any mismatch is silent — we trust the encoder's output.

        # 3. Build prompt token IDs (audio_pad x N tokens, plus forced-lang)
        prompt_ids, audio_offset = self._build_prompt(
            n_audio_tokens, language=language
        )

        # 4. Prefill
        input_ids = np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1)
        position_ids = np.arange(
            input_ids.shape[1], dtype=np.int64
        ).reshape(1, -1)
        audio_offset_arr = np.asarray([audio_offset], dtype=np.int64)

        logits, past_keys, past_values = self.decoder_init_sess.run(
            None,
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "audio_features": audio_features.astype(np.float32, copy=False),
                "audio_offset": audio_offset_arr,
            },
        )

        # 5. Greedy decode loop
        generated: list[int] = []
        cur_pos = int(input_ids.shape[1])
        next_id = int(np.argmax(logits[0, -1, :]))

        for _ in range(max_new_tokens):
            if next_id in EOS_TOKEN_IDS:
                break
            generated.append(next_id)

            # Embedding lookup (FP16 -> FP32 cast for the tiny slice only).
            embed = self._embed_tokens_fp16[next_id].astype(
                np.float32, copy=False
            ).reshape(1, 1, self.hidden_size)
            pos = np.asarray([[cur_pos]], dtype=np.int64)

            logits, past_keys, past_values = self.decoder_step_sess.run(
                None,
                {
                    "input_embeds": embed,
                    "position_ids": pos,
                    "past_keys": past_keys,
                    "past_values": past_values,
                },
            )
            next_id = int(np.argmax(logits[0, -1, :]))
            cur_pos += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = text.strip()

        # If a language was forced we inlined it in the prompt, so the model
        # only emits the transcription — no extra parsing needed.
        lang_out = language or ""
        return AsrResult(text=text, language=lang_out)

    # -------------------- internals ------------------------------------

    def _build_prompt(
        self,
        n_audio_tokens: int,
        language: Optional[str],
    ) -> Tuple[list[int], int]:
        """
        Construct the prompt token sequence and return (ids, audio_offset).
        audio_offset is the index of the first <|audio_pad|> token — needed
        by decoder_init.onnx to scatter audio features in place.
        """
        # Fixed prefix (system + user_start + audio_start) — 9 tokens.
        prefix = [
            IM_START_TOKEN_ID, SYSTEM_TOKEN_ID, NEWLINE_TOKEN_ID,
            IM_END_TOKEN_ID, NEWLINE_TOKEN_ID,
            IM_START_TOKEN_ID, USER_TOKEN_ID, NEWLINE_TOKEN_ID,
            AUDIO_START_TOKEN_ID,
        ]
        audio_offset = len(prefix)

        ids = list(prefix)
        ids.extend([AUDIO_PAD_TOKEN_ID] * n_audio_tokens)
        ids.extend([
            AUDIO_END_TOKEN_ID, IM_END_TOKEN_ID, NEWLINE_TOKEN_ID,
            IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID,
        ])

        if language:
            ids.extend(self._lang_prefix_ids(language))

        return ids, audio_offset

    def _lang_prefix_ids(self, language: str) -> list[int]:
        """
        Encode 'language {Name}' and append <asr_text> — matches exactly what
        qwen_asr's _build_text_prompt does when force_language is set.
        """
        key = language.strip()
        if key in self._lang_prefix_cache:
            return self._lang_prefix_cache[key]
        # Do NOT add special tokens — we're appending raw text to the prompt.
        enc = self.tokenizer.encode(f"language {key}", add_special_tokens=False)
        ids = list(enc.ids) + [ASR_TEXT_TOKEN_ID]
        self._lang_prefix_cache[key] = ids
        return ids

    # Allow the object to be used as a context manager (optional).
    def close(self) -> None:
        self.encoder_sess = None   # type: ignore[assignment]
        self.decoder_init_sess = None   # type: ignore[assignment]
        self.decoder_step_sess = None   # type: ignore[assignment]


# ----------------------------------------------------------------------------
# CLI smoke test: `python qwen_onnx_backend.py some.wav [--size 1.7B] [--lang English]`
# ----------------------------------------------------------------------------

def _main() -> int:
    import argparse
    import time
    import wave

    parser = argparse.ArgumentParser(description="Qwen3-ASR ONNX smoke test")
    parser.add_argument("audio", help="Path to a 16-bit mono WAV file")
    parser.add_argument("--size", choices=list(ONNX_REPO_MAP), default="0.6B")
    parser.add_argument("--lang", default=None, help="e.g. English, Hindi")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    print(f"[Info] Downloading / locating {ONNX_REPO_MAP[args.size]} ...")
    model_dir = download_onnx_model(args.size)
    print(f"[Info] Model dir: {model_dir}")

    pipe = Qwen3AsrOnnx(model_dir, num_threads=args.threads)

    with wave.open(args.audio, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print("[Error] Expected 16-bit mono WAV", file=sys.stderr)
            return 1
        sr = wf.getframerate()
        if sr != SAMPLE_RATE:
            print(f"[Error] Expected 16 kHz, got {sr}", file=sys.stderr)
            return 1
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    t0 = time.time()
    result = pipe.transcribe(samples, language=args.lang)
    dt = time.time() - t0
    print(f"[Info] Transcribed in {dt:.2f}s ({len(samples)/SAMPLE_RATE:.1f}s audio)")
    if result.language:
        print(f"[Info] Language: {result.language}")
    print(f"\n{result.text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
