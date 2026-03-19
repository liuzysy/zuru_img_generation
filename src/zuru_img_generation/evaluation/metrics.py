from __future__ import annotations

from pathlib import Path
import json
from PIL import Image
import numpy as np
from zuru_img_generation.utils.image_ops import compute_mask_iou, naive_foreground_mask


def hausdorff_distance(mask_a: Image.Image, mask_b: Image.Image) -> float:
    a = np.argwhere(np.array(mask_a.convert("L")) > 127)
    b = np.argwhere(np.array(mask_b.convert("L")) > 127)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    from scipy.spatial.distance import cdist
    dists = cdist(a, b)
    return float(max(dists.min(axis=1).max(), dists.min(axis=0).max()))


def evaluate_generated_dir(config: dict) -> dict:
    data_cfg = config["data"]
    infer_cfg = config["inference"]
    generated_dir = Path(data_cfg["generated_dir"])
    canonical_dir = Path(data_cfg["processed_dir"]) / "canonical"

    rows = []
    for p in sorted(generated_dir.glob("*.png")):
        shape_id = p.stem.split("__", 1)[0]
        target_mask_path = canonical_dir / f"{shape_id}_mask.png"
        if not target_mask_path.exists():
            continue
        gen = Image.open(p).convert("RGB")
        gen_mask = naive_foreground_mask(gen)
        target = Image.open(target_mask_path).convert("L")
        rows.append(
            {
                "file": p.name,
                "shape_id": shape_id,
                "iou": compute_mask_iou(gen_mask, target),
                "hausdorff": hausdorff_distance(gen_mask, target),
            }
        )

    summary = {
        "num_images": len(rows),
        "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
        "mean_hausdorff": float(np.mean([r["hausdorff"] for r in rows])) if rows else 0.0,
        "pass_rate": float(np.mean([r["iou"] >= float(infer_cfg["iou_threshold"]) for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    out = generated_dir / "evaluation_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
