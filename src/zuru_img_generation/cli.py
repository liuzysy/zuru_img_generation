from __future__ import annotations

import argparse
from pathlib import Path

from zuru_img_generation.config import load_config
from zuru_img_generation.data.preprocess import run_preprocess
from zuru_img_generation.training.lora_train import train_lora
from zuru_img_generation.inference.pipeline import generate_with_shape
from zuru_img_generation.evaluation.metrics import evaluate_generated_dir
from zuru_img_generation.utils.io import ensure_dir


def cmd_preprocess(args):
    cfg = load_config(args.config).raw
    result = run_preprocess(cfg)
    print(result)


def cmd_train(args):
    cfg = load_config(args.config).raw
    path = train_lora(cfg)
    print(path)


def cmd_infer(args):
    cfg = load_config(args.config).raw
    result = generate_with_shape(
        cfg,
        shape_id=args.shape_id,
        user_prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )
    out_dir = ensure_dir(cfg["data"]["generated_dir"])
    out_path = Path(out_dir) / f"{args.shape_id}__sample.png"
    result.image.save(out_path)
    print({
        "output": str(out_path),
        "accepted": result.accepted,
        "iou": result.iou,
        "final_prompt": result.final_prompt,
    })


def cmd_eval(args):
    cfg = load_config(args.config).raw
    print(evaluate_generated_dir(cfg))


def build_parser():
    parser = argparse.ArgumentParser(prog="zuru-img")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("preprocess")
    p1.add_argument("--config", required=True)
    p1.set_defaults(func=cmd_preprocess)

    p2 = sub.add_parser("train")
    p2.add_argument("--config", required=True)
    p2.set_defaults(func=cmd_train)

    p3 = sub.add_parser("infer")
    p3.add_argument("--config", required=True)
    p3.add_argument("--shape-id", required=True)
    p3.add_argument("--prompt", required=True)
    p3.add_argument("--steps", type=int, default=None)
    p3.add_argument("--guidance", type=float, default=None)
    p3.set_defaults(func=cmd_infer)

    p4 = sub.add_parser("eval")
    p4.add_argument("--config", required=True)
    p4.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
