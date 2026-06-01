"""Deterministic image quality gate.

Runs server-side BEFORE any LLM call. Rejects unusable uploads with a clear
reason so we never spend an API call on garbage input. Every rejection is
logged with the reason, the measured statistic, and the threshold so we can
tune the thresholds from real data.

Pipeline:

  decode --> EXIF auto-rotate --> dimension check --> blur check
         --> exposure check --> resize (longest edge 1024px) --> JPEG re-encode

The re-encode step strips EXIF (privacy + smaller payload) and normalizes the
output to image/jpeg regardless of input format.

Pillow + NumPy only — OpenCV would add ~200 MB to the Docker image for
operations Pillow handles natively.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

# Tuned by intuition; revise from real rejection logs once we have any.
MIN_DIMENSION_PX = 224
MAX_DIMENSION_PX = 1024
JPEG_QUALITY = 85
BLUR_VARIANCE_THRESHOLD = 80.0  # Laplacian variance; phone camera typical: 200-2000
DARK_LUMINANCE_THRESHOLD = 25.0  # mean grayscale 0-255
BRIGHT_LUMINANCE_THRESHOLD = 235.0


class RejectReason(StrEnum):
    NOT_DECODABLE = "not_decodable"
    TOO_SMALL = "too_small"
    TOO_BLURRY = "too_blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"


_USER_FACING = {
    RejectReason.NOT_DECODABLE: (
        "I couldn't read that image. Try a JPG or PNG photo of a single leaf."
    ),
    RejectReason.TOO_SMALL: (
        "That image is too small to diagnose. Please upload a photo at least "
        f"{MIN_DIMENSION_PX}px on each side."
    ),
    RejectReason.TOO_BLURRY: (
        "That photo looks blurry. Hold steady, tap to focus on the leaf, and try again."
    ),
    RejectReason.TOO_DARK: (
        "That photo is too dark. Move to better daylight (or use a lamp) and retake."
    ),
    RejectReason.TOO_BRIGHT: (
        "That photo is overexposed. Avoid direct sunlight on the leaf and retake."
    ),
}


@dataclass(frozen=True)
class GateAccept:
    """Image passed all checks. `data` is JPEG-encoded; `mime` is image/jpeg."""

    data: bytes
    mime: str
    width: int
    height: int


@dataclass(frozen=True)
class GateReject:
    """Image was rejected. `message` is safe to show to the user."""

    reason: RejectReason
    message: str


GateResult = GateAccept | GateReject


def gate(raw: bytes, mime: str | None) -> GateResult:
    """Run the quality gate on raw image bytes.

    Logs the outcome at INFO (accept) or WARNING (reject) so threshold tuning
    is data-driven, not vibes-driven.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        if img is None:
            return _reject(RejectReason.NOT_DECODABLE, mime=mime)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return _reject(RejectReason.NOT_DECODABLE, mime=mime)

    width, height = img.size
    if min(width, height) < MIN_DIMENSION_PX:
        return _reject(
            RejectReason.TOO_SMALL,
            mime=mime,
            extra=f"width={width} height={height}",
        )

    gray = np.asarray(img.convert("L"), dtype=np.float32)
    luminance = float(gray.mean())
    if luminance < DARK_LUMINANCE_THRESHOLD:
        return _reject(
            RejectReason.TOO_DARK,
            mime=mime,
            extra=f"luminance={luminance:.1f} threshold={DARK_LUMINANCE_THRESHOLD}",
        )
    if luminance > BRIGHT_LUMINANCE_THRESHOLD:
        return _reject(
            RejectReason.TOO_BRIGHT,
            mime=mime,
            extra=f"luminance={luminance:.1f} threshold={BRIGHT_LUMINANCE_THRESHOLD}",
        )

    blur_variance = _laplacian_variance(img)
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        return _reject(
            RejectReason.TOO_BLURRY,
            mime=mime,
            extra=f"variance={blur_variance:.1f} threshold={BLUR_VARIANCE_THRESHOLD}",
        )

    if max(width, height) > MAX_DIMENSION_PX:
        scale = MAX_DIMENSION_PX / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        width, height = img.size

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    out = buf.getvalue()

    log.info(
        "preprocess.accept input_mime=%s output_bytes=%d width=%d height=%d "
        "luminance=%.1f blur_variance=%.1f",
        mime,
        len(out),
        width,
        height,
        luminance,
        blur_variance,
    )
    return GateAccept(data=out, mime="image/jpeg", width=width, height=height)


def _laplacian_variance(img: Image.Image) -> float:
    """Variance of the Laplacian — canonical sharpness metric.

    Higher = sharper. Reference points: flat color ~0; heavily blurred photo
    20-80; sharp phone photo 300-2000.

    We compute the convolution in float32 with NumPy because Pillow's
    `ImageFilter.Kernel` clips output to uint8, which silently zeros out the
    negative half of the Laplacian response and turns this metric into noise.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _reject(reason: RejectReason, *, mime: str | None, extra: str = "") -> GateReject:
    log.warning("preprocess.reject reason=%s input_mime=%s %s", reason.value, mime, extra)
    return GateReject(reason=reason, message=_USER_FACING[reason])
