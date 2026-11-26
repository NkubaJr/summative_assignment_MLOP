"""Image preprocessing shared by the notebook, training, retraining and the API.

Each ECG sheet is 2213 x 1572 pixels. Shrinking the whole page to a small square wipes
out the thin heart trace and keeps the printed header, so instead we:
  1. crop to the ECG grid (drops the patient header and footer text),
  2. keep only the dark trace pixels (drops the pink grid),
  3. remove the leftover grid dots and thicken the trace so it survives resizing,
  4. resize with area interpolation to TRACE_SIZE,
  5. describe the trace with HOG (histogram of oriented gradients) features.
"""
import io
import os
import re
import uuid

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

from src import config

ALLOWED_EXT = (".jpg", ".jpeg", ".png")

# Crop box as fractions of the page (top, bottom, left, right)
CROP = (0.185, 0.89, 0.03, 0.985)
INK_THRESHOLD = 80
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


# ---------------------------------------------------------------- loading
def load_rgb_from_bytes(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def load_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def is_valid_image(data):
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def list_images(folder):
    """Return (paths, labels) for folder/<class>/<image>."""
    paths, labels = [], []
    if not os.path.isdir(folder):
        return paths, labels
    for cls in sorted(os.listdir(folder)):
        cdir = os.path.join(folder, cls)
        if not os.path.isdir(cdir):
            continue
        for name in sorted(os.listdir(cdir)):
            if name.lower().endswith(ALLOWED_EXT):
                paths.append(os.path.join(cdir, name))
                labels.append(cls)
    return paths, labels


# ---------------------------------------------------------------- trace extraction
def trace_mask(rgb):
    """Full-resolution binary mask (0/1) of the ECG trace inside the grid area."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape
    t, b, l, r = CROP
    gray = gray[int(h * t):int(h * b), int(w * l):int(w * r)]
    ink = (gray < INK_THRESHOLD).astype(np.uint8)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def clean_trace(rgb, size=None):
    """Cleaned, thickened trace resized to `size` (h, w). Values 0-255, trace is bright."""
    size = size or config.TRACE_SIZE
    ink = trace_mask(rgb) * 255
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8))
    return cv2.resize(ink, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def hog_features(trace):
    return hog(trace, **HOG_PARAMS).astype(np.float32)


def features_from_rgb(rgb):
    return hog_features(clean_trace(rgb))


def features_from_bytes(data):
    return features_from_rgb(load_rgb_from_bytes(data))


def features_from_paths(paths, cache_path=None, progress=None):
    """HOG features for many files. Features are cached on disk by path and modified time,
    so retraining only processes images it has not seen before."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                cache = dict(zip(z["keys"].tolist(), z["feats"]))
        except Exception:
            cache = {}
    out, changed = [], False
    for i, p in enumerate(paths):
        key = f"{os.path.abspath(p)}|{os.path.getmtime(p):.0f}"
        if key not in cache:
            cache[key] = features_from_rgb(load_rgb(p))
            changed = True
        out.append(cache[key])
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, len(paths))
    if cache_path and changed:
        keys = list(cache.keys())
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys), feats=np.stack([cache[k] for k in keys]))
        os.replace(tmp, cache_path)
    return np.stack(out) if out else np.zeros((0, 0), np.float32)


# ---------------------------------------------------------------- uploads
def save_upload(label, filename, data):
    """Stage an uploaded image under data/uploads/<label>/ with a unique, safe name."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename or "image.jpg"))
    name = f"{uuid.uuid4().hex[:8]}_{safe}"
    folder = os.path.join(config.UPLOAD_DIR, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------- features for visualizations
def describe_image(path):
    """Interpretable per-image features used in the notebook and the UI visualizations."""
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    mask = trace_mask(rgb)
    mh, mw = mask.shape
    # Bottom rhythm strip (lead II over ~10 s); skip the calibration pulse on the left
    strip = mask[int(mh * 0.80):, int(mw * 0.05):]
    cols = strip.any(axis=0)
    if cols.any():
        top = strip.argmax(axis=0)
        height = (strip.shape[0] - top)[cols].astype(float)
        amplitude = float(np.percentile(height, 99) - np.median(height))
    else:
        amplitude = 0.0
    return {
        "width": int(w),
        "height": int(h),
        "trace_ink_pct": float(mask.mean() * 100),   # % of the grid area covered by trace
        "rhythm_amplitude_px": amplitude,           # tallest deflection above baseline in lead II strip
    }
