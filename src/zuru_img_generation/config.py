from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
