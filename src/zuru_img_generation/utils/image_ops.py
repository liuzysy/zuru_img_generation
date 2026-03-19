from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_square(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def naive_foreground_mask(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(mask).convert("L")


def contour_from_mask(mask: Image.Image) -> Image.Image:
    arr = np.array(mask)
    edges = cv2.Canny(arr, 50, 150)
    return Image.fromarray(edges).convert("L")


def align_mask_to_center(mask: Image.Image, size: int) -> Image.Image:
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    canvas = Image.new("L", (size, size), 0)
    if len(xs) == 0 or len(ys) == 0:
        return canvas
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = Image.fromarray(arr[y0:y1+1, x0:x1+1]).convert("L")
    crop.thumbnail((int(size * 0.8), int(size * 0.8)))
    ox = (size - crop.width) // 2
    oy = (size - crop.height) // 2
    canvas.paste(crop, (ox, oy))
    return canvas


def aggregate_masks(mask_paths: list[str | Path], size: int) -> Image.Image:
    if not mask_paths:
        return Image.new("L", (size, size), 0)
    stack = []
    for p in mask_paths:
        m = Image.open(p).convert("L")
        m = align_mask_to_center(m, size)
        stack.append(np.array(m, dtype=np.float32) / 255.0)
    mean_mask = np.mean(np.stack(stack, axis=0), axis=0)
    out = (mean_mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(out).convert("L")


def compute_mask_iou(mask_a: Image.Image, mask_b: Image.Image, threshold: int = 127) -> float:
    a = np.array(mask_a.convert("L")) > threshold
    b = np.array(mask_b.convert("L")) > threshold
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)
