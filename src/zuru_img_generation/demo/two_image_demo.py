from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from PIL import Image
import numpy as np

from zuru_img_generation.utils.io import ensure_dir
from zuru_img_generation.utils.prompting import build_training_caption
from zuru_img_generation.utils.image_ops import resize_square


PAIR_FILES = {
    "pair_01": {
        "real": Path("dataset/two_image_demo/pair_01_real_sample.png"),
        "edge": Path("dataset/two_image_demo/pair_01_control_edge.png"),
        "shape": Path("dataset/two_image_demo/pair_01_shape_reference.png"),
    },
    "pair_02": {
        "real": Path("dataset/two_image_demo/pair_02_real_sample.png"),
        "edge": Path("dataset/two_image_demo/pair_02_control_edge.png"),
        "shape": Path("dataset/two_image_demo/pair_02_shape_reference.png"),
    },
}

DEFAULT_PAIR_PROMPTS = {
    "pair_01": "black furry plush toy, black shaggy fur all over body, oval upright body, sleepy half-closed droopy eyes, tiny glossy black nose, large open pink mouth with soft lips, a few small uneven teeth, weird ugly-cute expression, front-facing studio product photo, white background",
    "pair_02": "white furry plush toy, rounded rectangular body, short side arms, short separated legs, small centered pink mouth with dark gray outline, tiny teeth, minimal or barely visible eyes, front-facing studio product photo, white background",
}


@dataclass
class TwoImageDemoBundle:
    work_dir: Path
    manifest_path: Path
    infer_record: dict
    records: list[dict]


def _binarize_shape_reference(src: Path, dst: Path, image_size: int) -> Path:
    img = Image.open(src).convert("L")
    img = resize_square(img.convert("RGB"), image_size).convert("L")
    arr = np.array(img)
    mask = (arr > 127).astype(np.uint8) * 255
    Image.fromarray(mask).save(dst)
    return dst


def build_two_image_demo_bundle(
    output_dir: str | Path,
    prompt_1: str | None = None,
    prompt_2: str | None = None,
    infer_pair: str = "pair_01",
    image_size: int = 1024,
) -> TwoImageDemoBundle:
    work_dir = ensure_dir(output_dir)
    records = []

    prompt_map = {
        "pair_01": prompt_1 or DEFAULT_PAIR_PROMPTS["pair_01"],
        "pair_02": prompt_2 or DEFAULT_PAIR_PROMPTS["pair_02"],
    }

    for pair_name, files in PAIR_FILES.items():
        pair_dir = ensure_dir(work_dir / pair_name)
        image_dst = pair_dir / "image.png"
        edge_dst = pair_dir / "edge.png"
        mask_dst = pair_dir / "mask.png"

        resize_square(Image.open(files["real"]).convert("RGB"), image_size).save(image_dst)
        resize_square(Image.open(files["edge"]).convert("RGB"), image_size).save(edge_dst)
        _binarize_shape_reference(files["shape"], mask_dst, image_size=image_size)

        user_prompt = prompt_map[pair_name]
        record = {
            "pair_name": pair_name,
            "shape_id": pair_name,
            "image_path": str(image_dst),
            "edge_path": str(edge_dst),
            "mask_path": str(mask_dst),
            "caption": build_training_caption(pair_name, user_prompt),
            "user_prompt": user_prompt,
        }
        records.append(record)

    manifest_path = work_dir / "two_image_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    infer_record = next(r for r in records if r["pair_name"] == infer_pair)
    return TwoImageDemoBundle(
        work_dir=work_dir,
        manifest_path=manifest_path,
        infer_record=infer_record,
        records=records,
    )
