from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from PIL import Image
from zuru_img_generation.utils.io import ensure_dir, write_json
from zuru_img_generation.utils.image_ops import (
    load_rgb,
    resize_square,
    naive_foreground_mask,
    contour_from_mask,
    aggregate_masks,
)
from zuru_img_generation.utils.prompting import build_training_caption


def run_preprocess(config: dict) -> dict:
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = ensure_dir(data_cfg["processed_dir"])
    image_dir = ensure_dir(processed_dir / "images")
    mask_dir = ensure_dir(processed_dir / "masks")
    edge_dir = ensure_dir(processed_dir / "edges")
    canonical_dir = ensure_dir(processed_dir / "canonical")
    image_size = int(data_cfg["image_size"])

    manifest = []
    per_shape_masks = defaultdict(list)

    for shape_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        shape_id = shape_dir.name
        for img_path in sorted(shape_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            stem = f"{shape_id}_{img_path.stem}"
            img = resize_square(load_rgb(img_path), image_size)
            mask = naive_foreground_mask(img)
            edge = contour_from_mask(mask)

            out_img = image_dir / f"{stem}.png"
            out_mask = mask_dir / f"{stem}.png"
            out_edge = edge_dir / f"{stem}.png"
            img.save(out_img)
            mask.save(out_mask)
            edge.save(out_edge)
            per_shape_masks[shape_id].append(out_mask)

            manifest.append(
                {
                    "shape_id": shape_id,
                    "image_path": str(out_img),
                    "mask_path": str(out_mask),
                    "edge_path": str(out_edge),
                    "caption": build_training_caption(shape_id),
                }
            )

    for shape_id, mask_paths in per_shape_masks.items():
        canonical_mask = aggregate_masks(mask_paths, image_size)
        canonical_edge = contour_from_mask(canonical_mask)
        canonical_mask.save(canonical_dir / f"{shape_id}_mask.png")
        canonical_edge.save(canonical_dir / f"{shape_id}_edge.png")

    write_json(processed_dir / "manifest.json", manifest)
    return {"num_samples": len(manifest), "num_shapes": len(per_shape_masks)}
