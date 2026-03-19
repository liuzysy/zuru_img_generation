from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import torch
from peft import PeftModel

from zuru_img_generation.training.model_factory import load_model_bundle
from zuru_img_generation.utils.image_ops import naive_foreground_mask, compute_mask_iou
from zuru_img_generation.utils.prompting import build_inference_prompt, detect_prompt_conflicts


@dataclass
class InferenceResult:
    image: Image.Image
    accepted: bool
    iou: float | None
    final_prompt: str


def load_pipeline(config: dict):
    model_cfg = config["model"]
    bundle = load_model_bundle(model_cfg, dtype=torch.float16)
    lora_path = model_cfg.get("lora_path")
    if lora_path and Path(lora_path).exists():
        bundle.pipeline.unet = PeftModel.from_pretrained(bundle.pipeline.unet, lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle.pipeline = bundle.pipeline.to(device)
    return bundle


def generate_with_control_image(
    config: dict,
    control_image_path: str | Path,
    user_prompt: str,
    shape_label: str = "selected shape",
    target_mask_path: str | Path | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    max_attempts: int | None = None,
) -> InferenceResult:
    control_image_path = Path(control_image_path)
    if not control_image_path.exists():
        raise FileNotFoundError(f"Control image not found: {control_image_path}")

    conflicts = detect_prompt_conflicts(user_prompt)
    final_prompt = build_inference_prompt(shape_label, user_prompt)
    if conflicts:
        final_prompt += ", preserve selected body shape over conflicting prompt geometry"

    infer_cfg = config["inference"]
    steps = num_inference_steps or int(infer_cfg["num_inference_steps"])
    gs = guidance_scale or float(infer_cfg["guidance_scale"])
    conditioning_scale = float(infer_cfg.get("controlnet_conditioning_scale", 0.9))
    negative_prompt = infer_cfg.get("negative_prompt", "")

    control_image = Image.open(control_image_path).convert("RGB")
    attempts = max(1, max_attempts if max_attempts is not None else int(infer_cfg.get("max_resamples", 1)))

    best_result = None
    best_iou = -1.0
    accepted = True
    iou = None
    target_mask = None

    if target_mask_path is not None:
        target_mask_path = Path(target_mask_path)
        if not target_mask_path.exists():
            raise FileNotFoundError(f"Target mask not found: {target_mask_path}")
        target_mask = Image.open(target_mask_path).convert("L")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for attempt in range(attempts):
        bundle = load_pipeline(config)
        pipe = bundle.pipeline
        generator = torch.Generator(device=device).manual_seed(1000 + attempt)

        result = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=steps,
            guidance_scale=gs,
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        ).images[0]

        if target_mask is None:
            best_result = result
            break

        current_iou = compute_mask_iou(naive_foreground_mask(result), target_mask)
        current_accepted = current_iou >= float(infer_cfg["iou_threshold"])
        if current_iou > best_iou:
            best_iou, best_result, iou, accepted = current_iou, result, current_iou, current_accepted
        if current_accepted:
            break

    return InferenceResult(image=best_result, accepted=accepted, iou=iou, final_prompt=final_prompt)
