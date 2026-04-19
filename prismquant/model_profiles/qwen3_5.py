"""Qwen3.5 / Qwen3.6 MoE profile.

Covers:
  - Qwen3_5MoeForConditionalGeneration (multimodal, MoE)
  - Qwen3_5MoeForCausalLM (text-only MoE)
  - Qwen3_5MoeTextModel  (headless)

The two naming conventions PrismQuant must juggle:

  | where                    | body                                         |
  |--------------------------|----------------------------------------------|
  | HF multimodal source     | model.language_model.layers.X.*              |
  | vLLM scheme-dispatch     | language_model.model.layers.X.*              |
  | HF text-only / lm_head   | lm_head                                      |
  | vLLM scheme-dispatch     | language_model.lm_head                       |
  | MTP source               | mtp.layers.0.*   (mtp.fc, mtp.norm, ...)     |
  | vLLM MTP scheme-dispatch | mtp.layers.0.*   (IDENTITY — mtp. → model.   |
  |                          |                    remap only at weight-load)|

Visual encoder blocks pass through as BF16 (no real calibration yet).
"""
from __future__ import annotations

import copy
import os
import re

import torch.nn as nn

from .base import ModelProfile


class Qwen3_5Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5"}:
            return True
        for arch in architectures:
            if arch.startswith("Qwen3_5") or arch.startswith("Qwen3.5") \
                    or arch.startswith("Qwen3_6") or arch.startswith("Qwen3.6"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5"

    # ------------------------------------------------------------
    # Fused-sibling promotion
    # ------------------------------------------------------------
    _FUSED_PATTERNS = (
        # Standard attention q/k/v/o
        (re.compile(r"^(.+?\.self_attn)\.(q|k|v|o)_proj$"),   r"\1.qkvo"),
        # MLP gate/up siblings
        (re.compile(r"^(.+?\.mlp)\.(gate|up)_proj$"),          r"\1.gate_up"),
        (re.compile(r"^(.+?\.shared_expert)\.(gate|up)_proj$"), r"\1.gate_up"),
        # Qwen3.6 GatedDeltaNet: in_proj_qkv + in_proj_z share one bmm in vLLM
        (re.compile(r"^(.+?\.linear_attn)\.in_proj_(qkv|z)$"),  r"\1.qkvz"),
        # Qwen3.6 GatedDeltaNet: in_proj_a + in_proj_b
        (re.compile(r"^(.+?\.linear_attn)\.in_proj_(a|b)$"),    r"\1.ab"),
    )

    def fused_sibling_group(self, linear_qname: str) -> str | None:
        for pat, repl in self._FUSED_PATTERNS:
            m = pat.match(linear_qname)
            if m:
                return pat.sub(repl, linear_qname)
        return None

    # ------------------------------------------------------------
    # MoE
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"gate_up_proj", "down_proj"})

    def per_expert_moe_regex(self) -> str | None:
        # vLLM constructs per-expert Linears under
        # language_model.model.layers.X.mlp.experts.Y.{gate|up|down}_proj
        return (r"re:^language_model[.]model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return True

    def per_expert_mtp_regex(self) -> str | None:
        # MTP prefix stays `mtp.` at scheme dispatch (Qwen3_5MTP passes
        # prefix="mtp" to its inner predictor). The `mtp.→model.` rewrite
        # only happens in the weight loader.
        return (r"re:^mtp[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def mtp_layer_count(self, cfg: dict) -> int:
        # Use base implementation first.
        n = super().mtp_layer_count(cfg)
        if n > 0:
            return n
        # Qwen3.6 uses `mtp_num_hidden_layers` on text_config. Covered
        # above. If still zero, scan safetensors as a last resort.
        return 0  # caller can scan safetensors separately if desired

    def build_mtp_module(self, text_config) -> nn.Module:
        """Mirror vLLM's `Qwen3_5MultiTokenPredictor.forward`:

            e = pre_fc_norm_embedding(inputs_embeds)
            h = pre_fc_norm_hidden(body_hidden_states)
            h = fc(cat([e, h]))
            h = layers[0](h, pos_embeddings, causal_mask)
            h = norm(h)
            return h

        Implemented on HF primitives so Fisher hooks and autograd work
        normally."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm,
        )
        mtp_cfg = _single_layer_full_attention_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        class _MtpModule(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.fc = nn.Linear(hidden * 2, hidden, bias=False)
                self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, layer_idx=0)])
                self.norm = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=eps)

            def forward(self, inputs_embeds, body_hidden_states,
                        position_embeddings, causal_mask, position_ids):
                e = self.pre_fc_norm_embedding(inputs_embeds)
                h = self.pre_fc_norm_hidden(body_hidden_states)
                h = torch.cat([e, h], dim=-1)
                h = self.fc(h)
                h = self.layers[0](
                    hidden_states=h,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                if isinstance(h, tuple):
                    h = h[0]
                h = self.norm(h)
                return h

        import torch
        return _MtpModule(mtp_cfg)

    def mtp_objective_example(self) -> str:
        return ("CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2}) — "
                "the aux-loss Qwen3.5/3.6 MTP was trained under.")

    # ------------------------------------------------------------
    # Naming remap for compressed-tensors scheme dispatch
    # ------------------------------------------------------------
    def to_vllm_internal_name(self, name: str) -> str:
        # Body — multimodal source form.
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model."):]
        # Visual encoder — vLLM maps `model.visual.` -> `visual.`.
        if name.startswith("model.visual."):
            return "visual." + name[len("model.visual."):]
        # MTP — preserve the `mtp.` prefix at scheme dispatch. vLLM's
        # `mtp.→model.` rewrite is a weight-loader transform that runs
        # AFTER scheme dispatch, so it does NOT affect target matching.
        if name.startswith("mtp."):
            return name
        # Body — text-only recipe form.
        if (name.startswith("model.layers.")
                or name.startswith("model.embed_tokens")
                or name.startswith("model.norm")
                or name == "model"):
            return "language_model.model." + name[len("model."):]
        # lm_head — qwen3_5 multimodal class maps under language_model.
        if name == "lm_head" or name.startswith("lm_head."):
            return "language_model." + name
        return name

    # ------------------------------------------------------------
    # Source passthrough + staging
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        # Visual encoder is passthrough (real calibration deferred); MTP
        # weights without a layer_config entry go through passthrough too.
        return ("model.visual.", "mtp.")

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        return (
            "vision_config", "audio_config", "speech_config",
            "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id",
        )

    # ------------------------------------------------------------
    # Shard prefixes
    # ------------------------------------------------------------
    def visual_layer_prefix(self) -> str:
        return "model.visual.blocks"

    def visual_config_key(self) -> str:
        return "vision_config"


def _single_layer_full_attention_config(text_config):
    """Shallow-copy the text config and pin it to one full-attention
    decoder layer — that's what vLLM's MTP uses."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg
