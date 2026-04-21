"""Default ModelProfile — used when no specific profile matches the model.

Minimal coverage:
  - No fused-sibling promotion (safest assumption — every Linear is
    independent).
  - No MTP support.
  - No visual encoder.
  - Identity name remap (`to_vllm_internal_name` returns input unchanged).
  - Generic packed-expert attribute names (`gate_up_proj`, `down_proj`,
    `w1`/`w2`/`w3`, `gate_proj`/`up_proj`) so Mixtral-style MoEs still
    work out of the box.

If you're adding a new architecture, start from this as a baseline and
override only what differs.
"""
from __future__ import annotations

from .base import ModelProfile


class DefaultProfile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        # DefaultProfile is the terminal fallback — never claims anything
        # affirmatively. `registry.detect_profile` picks it when every
        # other profile returns False.
        return False

    @property
    def name(self) -> str:
        return "default"
