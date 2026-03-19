from __future__ import annotations

CONFLICT_TERMS = {
    "long legs": ["stubby", "short body"],
    "thin body": ["round body", "fat body"],
    "tiny head": ["large head"],
    "very tall body": ["short plush toy"],
}


def build_training_caption(shape_id: str, user_caption: str | None = None) -> str:
    if user_caption:
        return user_caption
    return f"Fuggler plush toy, {shape_id}, soft fluffy material, white background"


def build_inference_prompt(shape_id: str, user_prompt: str) -> str:
    return user_prompt


def detect_prompt_conflicts(user_prompt: str) -> list[str]:
    prompt_lower = user_prompt.lower()
    hits = []
    for key, values in CONFLICT_TERMS.items():
        if key in prompt_lower:
            hits.append(key)
        for value in values:
            if value in prompt_lower:
                hits.append(value)
    return sorted(set(hits))
