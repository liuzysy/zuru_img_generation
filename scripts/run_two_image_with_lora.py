from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

from zuru_img_generation.config import load_config
from zuru_img_generation.demo.two_image_demo import build_two_image_demo_bundle, DEFAULT_PAIR_PROMPTS
from zuru_img_generation.inference.pipeline import generate_with_control_image
from zuru_img_generation.training.lora_train import train_lora


def main():
    parser = argparse.ArgumentParser(description="Train LoRA on two image/control/mask/caption pairs and then run inference with the saved LoRA to reproduce similar results.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--prompt-1", default=DEFAULT_PAIR_PROMPTS["pair_01"])
    parser.add_argument("--prompt-2", default=DEFAULT_PAIR_PROMPTS["pair_02"])
    parser.add_argument("--infer-pair", choices=["pair_01", "pair_02"], default="pair_01")
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--skip-iou-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    bundle = build_two_image_demo_bundle(
        output_dir="outputs/two_image_with_lora",
        prompt_1=args.prompt_1,
        prompt_2=args.prompt_2,
        infer_pair=args.infer_pair,
        image_size=int(cfg["data"]["image_size"]),
    )

    cfg["training"]["output_dir"] = str(bundle.work_dir)
    cfg["training"]["epochs"] = 1000
    cfg["training"]["batch_size"] = 1
    cfg["training"]["max_steps"] = args.train_steps
    cfg["model"]["lora_path"] = str(bundle.work_dir / "lora")

    lora_path = train_lora(cfg, smoke_manifest_path=bundle.manifest_path, max_steps=args.train_steps)

    record = bundle.infer_record
    target_mask = None if args.skip_iou_check else record["mask_path"]
    result = generate_with_control_image(
        cfg,
        control_image_path=record["mask_path"],
        target_mask_path=target_mask,
        shape_label=record["shape_id"],
        user_prompt=record["user_prompt"],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )

    out_path = bundle.work_dir / f"{args.infer_pair}_generated.png"
    crop_size = 573
    img = result.image
    left = (img.width - crop_size) // 2
    top = (img.height - crop_size) // 2
    img.crop((left, top, left + crop_size, top + crop_size)).save(out_path)

    print({
        "mode": "two_image_with_lora_only",
        "manifest": str(bundle.manifest_path),
        "lora_path": lora_path,
        "output": str(out_path),
        "accepted": result.accepted,
        "iou": result.iou,
        "final_prompt": result.final_prompt,
        "used_control": record["mask_path"],
        "used_mask": target_mask,
        "train_steps": args.train_steps,
    })


if __name__ == "__main__":
    main()
