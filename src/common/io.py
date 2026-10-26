"""Image I/O helpers.

All images are handled as single-channel float32 arrays in [0, 1].
Reading/writing goes through np.fromfile / cv2.imdecode so that paths with
non-ASCII characters also work on Windows (cv2.imread does not).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert an integer/float image to float32 in [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        # 12-bit data (e.g. raw JSRT) is often stored in 16-bit containers.
        denom = 4095.0 if img.max() <= 4095 else 65535.0
        return img.astype(np.float32) / denom
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo + 1e-8)


def read_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM chest X-ray as float32 [0, 1] (VOI LUT + MONOCHROME1 fix)."""
    import pydicom

    try:  # pydicom >= 3
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1.0 - arr
    return arr


def imread_gray(path: str | Path) -> np.ndarray:
    """Read any supported image (PNG/JPG/TIFF/DICOM .dcm/.dicom) as grayscale float32 [0, 1]."""
    path = str(path)
    if path.lower().endswith((".dcm", ".dicom")):
        return read_dicom(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return to_float01(img)


def imwrite(path: str | Path, img: np.ndarray, bit16: bool = False) -> None:
    """Write a float [0, 1] image (gray HxW or color HxWx3, BGR order) as PNG/JPG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0.0, 1.0)
    if bit16:
        out = (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (img * 255.0 + 0.5).astype(np.uint8)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def resize(img: np.ndarray, size: tuple[int, int] | int) -> np.ndarray:
    """Resize to (width, height). A single int means a square image."""
    if isinstance(size, int):
        size = (size, size)
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (size[0] < w or size[1] < h) else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on a float [0, 1] image (computed on 16-bit to keep precision)."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    u16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    return op.apply(u16).astype(np.float32) / 65535.0


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path
