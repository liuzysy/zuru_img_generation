from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from zuru_img_generation.data.dataset import FugglerTrainingDataset, SimpleImageControlDataset
from zuru_img_generation.training.model_factory import load_model_bundle
from zuru_img_generation.utils.io import ensure_dir


class TrainingNotReadyError(RuntimeError):
    pass


def _attach_lora(module, rank: int):
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(module, lora_cfg)


def _build_dataset(config: dict, smoke_manifest_path: str | Path | None = None):
    if smoke_manifest_path is not None:
        with Path(smoke_manifest_path).open("r", encoding="utf-8") as f:
            items = json.load(f)
        return SimpleImageControlDataset(items)
    manifest_path = Path(config["data"]["processed_dir"]) / "manifest.json"
    if not manifest_path.exists():
        raise TrainingNotReadyError("manifest.json not found.")
    return FugglerTrainingDataset(manifest_path)


def train_lora(config: dict, smoke_manifest_path: str | Path | None = None, max_steps: int | None = None) -> str:
    train_cfg = config["training"]
    model_cfg = config["model"]

    accelerator = Accelerator(mixed_precision=train_cfg.get("mixed_precision", "bf16"))
    device = accelerator.device
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16

    bundle = load_model_bundle(model_cfg, dtype=dtype)
    bundle.denoiser = _attach_lora(bundle.denoiser, rank=int(train_cfg["lora_rank"]))
    pipe = bundle.pipeline
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    dataset = _build_dataset(config, smoke_manifest_path=smoke_manifest_path)
    loader = DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        bundle.denoiser.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    bundle.denoiser, optimizer, loader = accelerator.prepare(bundle.denoiser, optimizer, loader)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    output_dir = ensure_dir(train_cfg["output_dir"])
    global_step = 0
    step_limit = max_steps if max_steps is not None else int(train_cfg.get("max_steps", 0) or 0)

    for epoch in range(int(train_cfg["epochs"])):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}", disable=not accelerator.is_local_main_process)
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            control_image = batch.get("control", batch["edge"]).repeat(1, 3, 1, 1).to(device=device, dtype=dtype)
            prompts = list(batch["caption"])

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompts, device=device, do_classifier_free_guidance=False
                )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": torch.zeros((latents.shape[0], 6), device=device, dtype=prompt_embeds.dtype),
            }
            down_samples, mid_sample = pipe.controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            model_pred = bundle.denoiser(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
                return_dict=False,
            )[0]

            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(step=global_step, loss=float(loss.detach().item()))
            if step_limit and global_step >= step_limit:
                break
        if step_limit and global_step >= step_limit:
            break

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(bundle.denoiser)
    save_dir = ensure_dir(output_dir / "lora")
    unwrapped.save_pretrained(save_dir)
    return str(save_dir)
