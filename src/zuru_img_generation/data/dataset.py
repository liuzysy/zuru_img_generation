from __future__ import annotations

from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import json


class FugglerTrainingDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            self.items: list[dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(item["mask_path"]).convert("L"), dtype=np.float32) / 255.0
        edge = np.array(Image.open(item["edge_path"]).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0)
        control = mask.clone()
        return {
            "pixel_values": image,
            "mask": mask,
            "edge": edge,
            "control": control,
            "caption": item["caption"],
            "shape_id": item["shape_id"],
        }


class SimpleImageControlDataset(Dataset):
    """A tiny dataset for smoke-testing LoRA training with a few direct image/control pairs."""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        edge = np.array(Image.open(item["edge_path"]).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        edge = torch.from_numpy(edge).unsqueeze(0)
        sample = {
            "pixel_values": image,
            "edge": edge,
            "caption": item["caption"],
            "shape_id": item.get("shape_id", "demo_shape"),
        }
        if item.get("mask_path"):
            mask = np.array(Image.open(item["mask_path"]).convert("L"), dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
            sample["mask"] = mask
            sample["control"] = mask.clone()
        else:
            sample["control"] = edge.clone()
        return sample
