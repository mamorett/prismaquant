"""Gemma 4 profile (Google's multimodal family — text + vision + audio).

Covers:
  - Gemma4ForConditionalGeneration (multimodal MoE + dense, all sizes)
  - Gemma4ForCausalLM (text-only)

Almost entirely vLLM-metadata-derived — Gemma 4 has a clean
`packed_modules_mapping` (`qkv_proj`, `gate_up_proj`) and a standard
`hf_to_vllm_mapper` that matches Qwen3.5/3.6's body-prefix convention.
No MTP heads (not in vLLM's speculative registry at this vLLM version),
so PrismQuant doesn't need a custom MTP forward builder.

Source passthrough prefixes cover the three modality towers (vision,
audio, and their embedding projectors) — these pass through as BF16
until we wire real multimodal calibration, matching the Qwen3.6 visual
encoder policy.

Minimal size: ~30 lines. Everything else inherits from base.
"""
from __future__ import annotations

from .base import ModelProfile


class Gemma4Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"gemma4", "gemma4_text"}:
            return True
        for arch in architectures:
            if arch.startswith("Gemma4"):
                return True
        return False

    @property
    def name(self) -> str:
        return "gemma4"

    def vllm_architecture_class(self) -> str:
        # `Gemma4ForConditionalGeneration` exposes the full multimodal
        # prefix map (vision_tower, audio_tower, embed_vision,
        # embed_audio, language_model). Auto-derived
        # `fused_sibling_group` and `to_vllm_internal_name` inherit
        # from base — no overrides needed.
        return "Gemma4ForConditionalGeneration"

    # ------------------------------------------------------------
    # MoE (only on MoE variants like gemma-4-26b-a4b)
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"gate_up_proj", "down_proj"})

    def per_expert_moe_regex(self) -> str | None:
        # vLLM constructs per-expert Linears under
        # language_model.model.layers.X.mlp.experts.Y.{gate|up|down}_proj
        # — same convention as Qwen3.5 MoE. Dense Gemma 4 variants
        # (31b-it) simply won't have any MoE tensors, so this regex
        # matches nothing on disk and the allocator produces a single
        # format group.
        return (r"re:^language_model[.]model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    # ------------------------------------------------------------
    # Source passthrough (multimodal towers stay BF16 for v1)
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return (
            "model.vision_tower.",
            "model.audio_tower.",
            "model.embed_vision.",
            "model.embed_audio.",
        )

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        return (
            "vision_config", "audio_config", "speech_config",
            "image_token_id", "video_token_id", "audio_token_id",
            "vision_start_token_id", "vision_end_token_id",
        )

    def visual_config_key(self) -> str:
        return "vision_config"

    def visual_layer_prefix(self) -> str:
        # Gemma 4's vision tower uses the HF naming
        # `model.vision_tower.vision_model.encoder.layers.X.*`.
        # We only use this prefix for the probe's extended shard
        # regexes; the actual probe path for visual blocks is
        # deferred pending real multimodal calibration.
        return "model.vision_tower.vision_model.encoder.layers"
