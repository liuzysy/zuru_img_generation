from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel


@dataclass
class ModelBundle:
    family: str
    pipeline: Any
    denoiser: Any
    control_module: Any
    text_components_frozen: list[Any]


def load_model_bundle(model_cfg: dict, dtype: torch.dtype) -> ModelBundle:
    controlnet = ControlNetModel.from_pretrained(model_cfg["controlnet_model"], torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_cfg["base_model"], controlnet=controlnet, torch_dtype=dtype,
    )
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)
    return ModelBundle(
        family="sdxl",
        pipeline=pipe,
        denoiser=pipe.unet,
        control_module=pipe.controlnet,
        text_components_frozen=[pipe.vae, pipe.text_encoder, pipe.text_encoder_2],
    )
