"""vLLM model-registry introspection.

vLLM already encodes most of the architecture-specific knowledge
PrismQuant needs — fused-sibling structure (`packed_modules_mapping`)
and HF→vLLM name remaps (`hf_to_vllm_mapper`) — as class attributes
on its model classes. A PrismQuant profile that points at the right
vLLM class therefore gets correct `fused_sibling_group()` and
`to_vllm_internal_name()` for free, with no hand-coded patterns to
drift when vLLM evolves.

This module is the adapter. `vllm_class_for_architecture(arch_name)`
looks up the class, and the helpers below convert its class-level
metadata into PrismQuant's expected return shapes.

If vLLM isn't importable (CPU-only dev machine, etc.), everything
here degrades gracefully — the helpers return None and profiles fall
back to whatever they hand-coded.
"""
from __future__ import annotations

import re
from typing import Callable


def vllm_class_for_architecture(arch_name: str):
    """Return the vLLM class registered for an HF `architectures[0]`
    string (e.g. 'Qwen3_5MoeForConditionalGeneration'). Uses vLLM's
    model registry. Returns None if vLLM is unavailable or the
    architecture isn't registered.

    Note: vLLM's registry lazy-imports model modules; we only
    materialize the class, which is cheap relative to instantiating
    any weights."""
    try:
        from vllm.model_executor.models import registry
    except Exception:
        return None
    try:
        # Variable API shape across vLLM versions. Try the several
        # interfaces it has exposed. Modern vLLM (>= 0.19) uses
        # `ModelRegistry.resolve_model_cls([arch]) -> (cls, name)`.
        if hasattr(registry, "ModelRegistry"):
            reg = registry.ModelRegistry
            if hasattr(reg, "resolve_model_cls"):
                try:
                    out = reg.resolve_model_cls([arch_name])
                    if isinstance(out, tuple):
                        out = out[0]
                    # Modern vLLM sometimes returns a _RegisteredModel
                    # stub (lazy-loader record) — look for the real
                    # class under its `.model_cls` or `.cls` attributes.
                    for attr in ("model_cls", "cls"):
                        real = getattr(out, attr, None)
                        if real is not None and isinstance(real, type):
                            return real
                    if isinstance(out, type):
                        return out
                except Exception:
                    pass
            if hasattr(reg, "inspect_model_cls"):
                try:
                    info = reg.inspect_model_cls(arch_name)
                    # Some versions return a dataclass with a .cls field.
                    for attr in ("cls", "model_cls"):
                        real = getattr(info, attr, None)
                        if isinstance(real, type):
                            return real
                except Exception:
                    pass
            if hasattr(reg, "load_model_cls"):
                try:
                    return reg.load_model_cls(arch_name)
                except Exception:
                    pass
        # Fallbacks: direct dict lookups into the registry's internal tables.
        import importlib
        for tbl_name in ("_TEXT_GENERATION_MODELS", "_MULTIMODAL_MODELS",
                         "_SPECULATIVE_DECODING_MODELS", "_VLLM_MODELS",
                         "_SUPPORTED_MODELS"):
            tbl = getattr(registry, tbl_name, None)
            if not isinstance(tbl, dict) or arch_name not in tbl:
                continue
            entry = tbl[arch_name]
            # Entries can be tuples (mod_path, cls_name) or _LazyRegisteredModel.
            mod_path, cls_name = None, None
            if isinstance(entry, tuple) and len(entry) == 2:
                mod_path, cls_name = entry
            else:
                for mp_attr in ("module_name", "mod_name"):
                    mod_path = getattr(entry, mp_attr, None) or mod_path
                for cn_attr in ("class_name", "cls_name"):
                    cls_name = getattr(entry, cn_attr, None) or cls_name
            if mod_path and cls_name:
                try:
                    mod = importlib.import_module(
                        f"vllm.model_executor.models.{mod_path}")
                    return getattr(mod, cls_name, None)
                except Exception:
                    continue
    except Exception:
        return None
    return None


def packed_modules_mapping_from_class(cls) -> dict[str, list[str]]:
    """Return `{fused: [siblings, ...]}` from a vLLM model class's
    `packed_modules_mapping` attribute. Falls back to {} if missing."""
    if cls is None:
        return {}
    pm = getattr(cls, "packed_modules_mapping", None)
    if not isinstance(pm, dict):
        return {}
    out: dict[str, list[str]] = {}
    for fused, siblings in pm.items():
        if isinstance(siblings, list) and siblings:
            out[fused] = list(siblings)
    return out


def hf_to_vllm_prefix_map_from_class(cls) -> dict[str, str]:
    """Return `{hf_prefix: vllm_prefix}` from a vLLM model class's
    `hf_to_vllm_mapper` attribute. Falls back to {} if missing or if
    the mapper doesn't use prefix-substitution (we skip regex/substr
    mappers because they need full regex engine support)."""
    if cls is None:
        return {}
    mapper = getattr(cls, "hf_to_vllm_mapper", None)
    if mapper is None:
        return {}
    prefix = getattr(mapper, "orig_to_new_prefix", None)
    if not isinstance(prefix, dict):
        return {}
    return dict(prefix)


def fused_sibling_matcher_from_packed_mapping(
        packed_mapping: dict[str, list[str]],
) -> Callable[[str], str | None]:
    """Given `packed_modules_mapping` (the vLLM class attribute),
    return a function `fn(linear_qname) -> canonical_group_key | None`
    suitable for use as `ModelProfile.fused_sibling_group()`.

    The canonical group key collapses all siblings that would be
    fused in vLLM to the same string, so the allocator treats them as
    a single promotion bucket. Example: for Qwen3.5
    `packed_modules_mapping['qkv_proj'] = ['q_proj','k_proj','v_proj']`,
    any of `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`
    map to `self_attn.qkv_proj`.

    Sibling list is compiled into a single regex with the parent
    captured so we can substitute the fused name and return a stable
    group key. Patterns are evaluated in registration order so a more
    specific match (e.g. `in_proj_qkv` before `in_proj_q`) wins.
    """
    # Sort siblings by descending length so longer matches bind first
    # (prevents `in_proj_b` accidentally matching inside `in_proj_ba`
    # if both were registered without anchors — vLLM's conventions
    # don't do this today, but defense in depth is cheap).
    compiled: list[tuple[re.Pattern, str]] = []
    for fused, siblings in packed_mapping.items():
        siblings_sorted = sorted(siblings, key=len, reverse=True)
        alt = "|".join(re.escape(s) for s in siblings_sorted)
        # `(.*\.)` captures the parent module qname up to the Linear.
        # `(?:{alt})$` matches one of the sibling leaf names.
        pat = re.compile(rf"^(.*\.)?(?:{alt})$")
        compiled.append((pat, fused))

    def match(qname: str) -> str | None:
        for pat, fused in compiled:
            m = pat.match(qname)
            if m:
                parent = m.group(1) or ""
                return f"{parent}{fused}"
        return None

    return match


def name_remapper_from_prefix_map(
        prefix_map: dict[str, str],
) -> Callable[[str], str]:
    """Given an HF→vLLM prefix map (vLLM's `WeightsMapper.orig_to_new_prefix`),
    return a function that applies the first matching prefix rewrite.

    Longest-prefix-wins so `model.language_model.` rebinds before
    `model.` would."""
    items = sorted(prefix_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    def remap(name: str) -> str:
        for src, dst in items:
            if name.startswith(src):
                return dst + name[len(src):]
        return name

    return remap
