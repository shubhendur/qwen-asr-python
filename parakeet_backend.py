#!/usr/bin/env python3
"""
NVIDIA Parakeet TDT 0.6B v3 inference backend.

Two flavours are supported behind a single class so the dictation app stays
small and switch-statement-free:

  * MLX backend  (Apple Silicon only)
        Repo:  mlx-community/parakeet-tdt-0.6b-v3
        Pkg :  parakeet-mlx
        Disk:  ~2.5 GB (BF16)
        RAM :  ~1.3–2 GB (unified memory)
        Speed: ~30–50x real-time on M1 Pro

  * ONNX INT8 backend (cross-platform CPU)
        Repo:  istupakov/parakeet-tdt-0.6b-v3-onnx (auto-fetched by onnx-asr)
        Pkg :  onnx-asr
        Disk:  ~671 MB
        RAM :  ~2 GB
        Speed: ~12–20x real-time on i7-1355U

Both produce English (and 24 other European-language) transcripts; the
model is **not** trained for Hindi/CJK — those languages are silently
declined in the dictation menu.

Usage:
    from parakeet_backend import ParakeetAsr
    asr = ParakeetAsr(flavour="mlx")          # or "onnx-int8"
    text = asr.transcribe(audio_float32_16k)
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000


# Language codes Parakeet v3 understands. Used only for input validation —
# the underlying packages handle language detection automatically.
SUPPORTED_LANGUAGES = {
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English",
    "Estonian", "Finnish", "French", "German", "Greek", "Hungarian",
    "Italian", "Latvian", "Lithuanian", "Maltese", "Polish", "Portuguese",
    "Romanian", "Russian", "Slovak", "Slovenian", "Spanish", "Swedish",
    "Ukrainian",
}


@dataclass
class AsrResult:
    text: str
    language: str = ""


# ----------------------------------------------------------------------------
# Repository identifiers — kept here so download_model.py can import them too.
# ----------------------------------------------------------------------------

MLX_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
ONNX_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"   # used by `onnx-asr`


# ----------------------------------------------------------------------------
# Capability checks
# ----------------------------------------------------------------------------

def is_apple_silicon() -> bool:
    """True when running on macOS arm64 (M1/M2/M3/M4)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def supported_flavours() -> list[str]:
    """Return the Parakeet flavours that *could* run on this machine.

    We don't import parakeet-mlx / onnx-asr here — the dictation menu calls
    this before the user has installed anything. The menu shows everything
    available; install errors are surfaced lazily on selection.
    """
    flavours = ["onnx-int8"]   # always cross-platform via CPU
    if is_apple_silicon():
        flavours.insert(0, "mlx")
    return flavours


# ----------------------------------------------------------------------------
# Public class — wraps either backend behind one `transcribe()` call.
# ----------------------------------------------------------------------------

class ParakeetAsr:
    """End-to-end Parakeet-TDT-v3 transcription with a uniform API."""

    def __init__(
        self,
        flavour: str = "onnx-int8",
        *,
        num_threads: int = 0,
        chunk_seconds: float = 0.0,
    ):
        """
        Args:
            flavour: "mlx" (Apple Silicon GPU) or "onnx-int8" (CPU).
            num_threads: ONNX-only; intra-op thread cap (0 = auto).
            chunk_seconds: MLX-only; for files > a few minutes set this to
                e.g. 120 for chunked transcription. Live mic recordings are
                always short, so the default 0 (single-shot) is fine.
        """
        self.flavour = flavour
        self.num_threads = num_threads
        self.chunk_seconds = chunk_seconds

        if flavour == "mlx":
            self._init_mlx()
        elif flavour == "onnx-int8":
            self._init_onnx_int8()
        else:
            raise ValueError(
                f"Unknown Parakeet flavour {flavour!r}. "
                f"Pick one of: {supported_flavours()}"
            )

    # ---------------- MLX ------------------------------------------------

    def _init_mlx(self) -> None:
        if not is_apple_silicon():
            raise RuntimeError(
                "Parakeet MLX backend requires Apple Silicon (M1/M2/M3/M4)."
            )
        try:
            from parakeet_mlx import from_pretrained  # noqa: PLC0415
            import mlx.core as mx                     # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "parakeet-mlx is not installed. Install with:\n"
                "  pip install parakeet-mlx\n"
                "Or run the bundled helper:\n"
                "  ./setup.sh --parakeet"
            ) from e

        self._mx = mx
        # BF16 is the right default on Apple Silicon — full FP16 support, half
        # the memory of FP32, and matches what the model was trained in.
        self._model = from_pretrained(MLX_REPO, dtype=mx.bfloat16)

    def _transcribe_mlx(self, audio: np.ndarray) -> AsrResult:
        # parakeet-mlx (>= 0.5) only exposes a path-based API on its top-level
        # `transcribe()` call. Live-mic audio is always a short numpy array,
        # so we serialise it to an in-memory WAV via tempfile, transcribe, and
        # delete. Overhead at 16 kHz mono float32 is < 5 ms for a 30 s clip.
        import tempfile                       # noqa: PLC0415
        import wave                           # noqa: PLC0415

        # Convert float32 [-1, 1] -> int16 PCM and write a real WAV file.
        # tempfile uses the OS-managed temp dir; we delete it ourselves so
        # Windows file-locking doesn't trip us up.
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        kwargs: dict = {}
        if self.chunk_seconds and self.chunk_seconds > 0:
            kwargs["chunk_duration"] = float(self.chunk_seconds)
            kwargs["overlap_duration"] = 15.0

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())

            result = self._model.transcribe(tmp_path, **kwargs)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # parakeet-mlx returns an AlignedResult with a `.text` attribute.
        text = getattr(result, "text", "") or ""
        lang = getattr(result, "language", "") or ""
        return AsrResult(text=text.strip(), language=lang)

    # ---------------- ONNX INT8 -----------------------------------------

    def _init_onnx_int8(self) -> None:
        try:
            import onnx_asr   # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "onnx-asr is not installed. Install with:\n"
                "  pip install onnx-asr onnxruntime\n"
                "Or run the bundled helper:\n"
                "  ./setup.sh --parakeet"
            ) from e

        # Tame OS thread allocation — onnx-asr does not currently expose a
        # session-options hook, so set the env vars *before* the first call.
        if self.num_threads and self.num_threads > 0:
            os.environ.setdefault("OMP_NUM_THREADS", str(self.num_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(self.num_threads))

        # The first load triggers a HuggingFace fetch (~671 MB total).
        self._model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", quantization="int8"
        )

    def _transcribe_onnx(self, audio: np.ndarray) -> AsrResult:
        # onnx-asr expects either a file path or a numpy float32 mono @ 16 kHz.
        text = self._model.recognize(audio)
        # Some versions return a dataclass with .text; normalise.
        if hasattr(text, "text"):
            text = text.text  # type: ignore[attr-defined]
        return AsrResult(text=str(text).strip(), language="")

    # ---------------- Public API ----------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: Optional[str] = None,   # accepted for API parity; unused
    ) -> AsrResult:
        """Transcribe a 1-D float32 mono 16 kHz numpy array."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        if language and language not in SUPPORTED_LANGUAGES:
            # Don't fail — just inform once. Parakeet ignores the hint anyway.
            print(
                f"[Parakeet] Warning: '{language}' is not in Parakeet v3's "
                "supported language list; output will be auto-detected."
            )

        if self.flavour == "mlx":
            return self._transcribe_mlx(audio)
        return self._transcribe_onnx(audio)


# ----------------------------------------------------------------------------
# Pre-flight downloader — used by download_model.py.
# ----------------------------------------------------------------------------

def download_parakeet(flavour: str) -> str:
    """Pre-fetch Parakeet weights for the chosen flavour. Returns the cache dir."""
    from huggingface_hub import snapshot_download   # noqa: PLC0415

    if flavour == "mlx":
        return snapshot_download(repo_id=MLX_REPO)
    if flavour == "onnx-int8":
        # onnx-asr only needs the INT8 graphs + vocab — keep the download
        # surgical so we don't pull the FP32 fallbacks (~2.5 GB).
        return snapshot_download(
            repo_id=ONNX_REPO,
            allow_patterns=[
                "config.json",
                "encoder-model.int8.onnx",
                "decoder_joint-model.int8.onnx",
                "nemo128.onnx",
                "vocab.txt",
            ],
        )
    raise ValueError(f"Unknown flavour {flavour!r}")
