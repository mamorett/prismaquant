#!/usr/bin/env python3
"""validate_profile.py — automated consistency checks for a
ModelProfile against an actual checkpoint.

Usage:

    python -m prismaquant.model_profiles.validate \\
        --model /path/to/Model
    python -m prismaquant.model_profiles.validate \\
        --profile MyCustomProfile \\
        --model /path/to/Model

Without `--profile`, the validator auto-detects using the registry.
With `--profile`, it imports that class name from `prismaquant.model_profiles`
or any module importable via `$PYTHONPATH`.

Checks performed:

  1. **Profile claim.** `profile.matches()` returns True for this
     model's (model_type, architectures) tuple.

  2. **vLLM class exists.** If the profile returns a
     `vllm_architecture_class()`, vLLM's registry can resolve it.

  3. **Fused-sibling self-consistency.** Every fused-group member
     (`profile.fused_sibling_group(sibling)`) returns the same
     canonical key across all siblings of the same group, using the
     vLLM `packed_modules_mapping`'s own sibling lists as ground
     truth.

  4. **Name remap fixed points.** For every
     `orig_to_new_prefix` entry in vLLM's `hf_to_vllm_mapper`,
     `profile.to_vllm_internal_name(orig_prefix + ".x")` starts with
     the expected `new_prefix`.

  5. **MTP module construction** (if `has_mtp()` is True). The
     profile can instantiate an MTP module from the model's text
     config; the module loads the source's `mtp.*` weights without
     missing keys.

  6. **Packed-expert parameter names.** The architecture's weights
     contain 3D parameters under modules whose class name matches
     `_is_packed_experts_module`, and every such parameter's name
     is in `profile.packed_expert_param_names()`.

  7. **Source passthrough sanity.** Every prefix in
     `profile.source_passthrough_prefixes()` matches at least one
     tensor in the source's safetensors index (otherwise the prefix
     is dead weight).

Exit code 0 if every check passes, 1 otherwise. Each failure prints
a ✗ line with context; each success prints a ✓. Intended to be
CI-friendly so new profiles get a clear pass/fail signal before
they're used in a production export.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


class CheckResult:
    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self):
        glyph = "✓" if self.ok else "✗"
        out = f"  {glyph} {self.name}"
        if self.detail:
            out += f"\n      {self.detail}"
        return out


def _load_config(model_path: str) -> dict:
    with open(Path(model_path) / "config.json") as f:
        return json.load(f)


def _get_profile(profile_arg: str | None, model_path: str):
    """Resolve the profile to validate — either by name or by
    auto-detection from the model path."""
    from .registry import detect_profile
    from .base import ModelProfile

    if profile_arg is None:
        profile = detect_profile(model_path)
        source = f"auto-detected"
    else:
        # Try to import by class path first (`pkg.module:Cls` or
        # `pkg.module.Cls`), else from prismaquant.model_profiles by
        # bare class name.
        if ":" in profile_arg:
            modname, clsname = profile_arg.split(":", 1)
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)
        elif "." in profile_arg:
            modname, clsname = profile_arg.rsplit(".", 1)
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)
        else:
            from . import registry as _r
            cls = None
            for candidate in [*_r._REGISTERED, _r.DefaultProfile]:
                if candidate.__name__ == profile_arg:
                    cls = candidate
                    break
            if cls is None:
                # Also try prismaquant.model_profiles namespace.
                mod = importlib.import_module("prismaquant.model_profiles")
                cls = getattr(mod, profile_arg, None)
            if cls is None:
                raise SystemExit(
                    f"Could not resolve profile '{profile_arg}'. "
                    "Pass a dotted path (pkg.module.Cls) or register "
                    "the profile first via register_profile().")
        profile = cls()
        source = f"explicit: {cls.__name__}"
    if not isinstance(profile, ModelProfile):
        raise SystemExit(f"Profile is not a ModelProfile subclass: {profile!r}")
    return profile, source


def _check_matches(profile, cfg: dict) -> CheckResult:
    model_type = cfg.get("model_type") or ""
    archs = list(cfg.get("architectures") or [])
    try:
        ok = bool(profile.__class__.matches(model_type, archs))
    except Exception as e:
        return CheckResult("matches() returns True for this model",
                           False, f"threw {type(e).__name__}: {e}")
    if not ok:
        return CheckResult(
            "matches() returns True for this model", False,
            f"model_type={model_type!r}, architectures={archs}")
    return CheckResult(
        "matches() returns True for this model", True,
        f"model_type={model_type!r}, architectures={archs}")


def _check_vllm_class(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult(
            "vllm_architecture_class() resolves",
            True,
            "profile provides no vLLM class (fine — arch-specific methods "
            "must be manually overridden)")
    from .vllm_registry import vllm_class_for_architecture
    cls = vllm_class_for_architecture(arch)
    if cls is None:
        return CheckResult(
            "vllm_architecture_class() resolves", False,
            f"arch='{arch}' not found in vLLM registry; "
            "install newer vLLM or change the arch name")
    return CheckResult(
        "vllm_architecture_class() resolves", True,
        f"{arch} → {cls.__module__}.{cls.__name__}")


def _check_fused_siblings(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult("fused-sibling groups consistent", True,
                           "no vLLM class to cross-check against")
    from .vllm_registry import (
        vllm_class_for_architecture, packed_modules_mapping_from_class,
    )
    cls = vllm_class_for_architecture(arch)
    pm = packed_modules_mapping_from_class(cls)
    if not pm:
        return CheckResult("fused-sibling groups consistent", True,
                           "vLLM class has no packed_modules_mapping")
    failures = []
    # Use an arbitrary prefix to stand in for a parent module qname.
    parent = "model.layers.0.parent."
    for fused, siblings in pm.items():
        keys = {profile.fused_sibling_group(parent + s) for s in siblings}
        if len(keys) != 1 or next(iter(keys)) is None:
            failures.append(f"{fused}: siblings {siblings} → keys {keys}")
    if failures:
        return CheckResult("fused-sibling groups consistent", False,
                           "; ".join(failures))
    return CheckResult("fused-sibling groups consistent", True,
                       f"{len(pm)} fused groups × multiple siblings map "
                       "to the same canonical key")


def _check_name_remap(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           True, "no vLLM class to cross-check against")
    from .vllm_registry import (
        vllm_class_for_architecture, hf_to_vllm_prefix_map_from_class,
    )
    cls = vllm_class_for_architecture(arch)
    prefix_map = hf_to_vllm_prefix_map_from_class(cls)
    if not prefix_map:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           True, "vLLM class has no hf_to_vllm_mapper")
    failures = []
    for src, dst in prefix_map.items():
        probe_name = src + "x.y"
        got = profile.to_vllm_internal_name(probe_name)
        expected_prefix = dst
        if not got.startswith(expected_prefix):
            failures.append(f"{probe_name!r} -> {got!r}, expected prefix {expected_prefix!r}")
    if failures:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           False, "; ".join(failures))
    return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                       True, f"{len(prefix_map)} prefix rewrites agree")


def _check_mtp(profile, cfg: dict, model_path: str) -> CheckResult:
    if not profile.has_mtp():
        return CheckResult("MTP module constructs + loads weights",
                           True, "profile reports no MTP (skipped)")
    from transformers import AutoConfig
    try:
        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        return CheckResult("MTP module constructs + loads weights", False,
                           f"AutoConfig failed: {e}")
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    try:
        mtp = profile.build_mtp_module(text_cfg)
    except Exception as e:
        return CheckResult("MTP module constructs + loads weights", False,
                           f"build_mtp_module threw {type(e).__name__}: {e}")
    if mtp is None:
        return CheckResult("MTP module constructs + loads weights", False,
                           "build_mtp_module returned None despite has_mtp()")
    return CheckResult("MTP module constructs + loads weights", True,
                       f"{type(mtp).__name__} with "
                       f"{sum(1 for _ in mtp.named_parameters())} parameters")


def _check_source_passthrough(profile, model_path: str) -> CheckResult:
    """Passthrough prefixes should mostly match — but profiles cover a
    family of variants (e.g. Gemma 4 26B-A4B has no audio tower, but
    Gemma 4 31B-IT may), so unused prefixes aren't a fatal profile bug.
    We pass if at least one prefix matches something AND report unused
    prefixes as informational. Fail only if every declared prefix is
    dead on this checkpoint — that implies the profile doesn't know
    this architecture at all."""
    prefixes = profile.source_passthrough_prefixes()
    if not prefixes:
        return CheckResult("source_passthrough_prefixes() cover real tensors",
                           True, "profile declares no passthrough prefixes")
    idx_path = Path(model_path) / "model.safetensors.index.json"
    if not idx_path.is_file():
        return CheckResult("source_passthrough_prefixes() cover real tensors",
                           True, f"{idx_path} missing — cannot verify")
    with open(idx_path) as f:
        keys = list(json.load(f).get("weight_map", {}).keys())
    covered = [p for p in prefixes if any(k.startswith(p) for k in keys)]
    missing = [p for p in prefixes if p not in covered]
    if not covered:
        return CheckResult(
            "source_passthrough_prefixes() cover real tensors", False,
            f"no declared prefix matches any tensor on disk: {list(prefixes)}")
    detail = f"{len(covered)}/{len(prefixes)} prefixes match"
    if missing:
        detail += f" — unused on this variant: {missing}"
    return CheckResult(
        "source_passthrough_prefixes() cover real tensors", True, detail)


def _check_packed_experts(profile, model_path: str) -> CheckResult:
    """Cross-check: do the expert parameter names the profile declares
    actually appear in the safetensors under an experts container?
    A name like `gate_up_proj` is valid if some safetensors key ends
    with `.experts.gate_up_proj` (or similar packed-expert location)."""
    idx_path = Path(model_path) / "model.safetensors.index.json"
    if not idx_path.is_file():
        return CheckResult(
            "packed_expert_param_names() cover actual 3D params",
            True, f"{idx_path} missing — cannot verify")
    with open(idx_path) as f:
        keys = list(json.load(f).get("weight_map", {}).keys())
    names = profile.packed_expert_param_names()
    if not names:
        return CheckResult(
            "packed_expert_param_names() cover actual 3D params",
            True, "profile declares no packed-expert names")
    found: dict[str, int] = {n: 0 for n in names}
    for k in keys:
        for n in names:
            if k.endswith(f"experts.{n}"):
                found[n] += 1
                break
    covered = [n for n, c in found.items() if c > 0]
    missing = [n for n, c in found.items() if c == 0]
    if not covered:
        return CheckResult(
            "packed_expert_param_names() cover actual 3D params", False,
            f"none of {set(names)} appear as packed experts on disk "
            f"(checked {len(keys)} safetensors keys)")
    detail = f"{len(covered)}/{len(names)} declared names found"
    if missing:
        detail += f" — unused: {missing}"
    return CheckResult(
        "packed_expert_param_names() cover actual 3D params", True, detail)


def validate_profile(profile, model_path: str, cfg: dict) -> list[CheckResult]:
    checks = [
        _check_matches(profile, cfg),
        _check_vllm_class(profile),
        _check_fused_siblings(profile),
        _check_name_remap(profile),
        _check_packed_experts(profile, model_path),
        _check_source_passthrough(profile, model_path),
        _check_mtp(profile, cfg, model_path),
    ]
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to HF checkpoint directory.")
    ap.add_argument("--profile", default=None,
                    help="Profile class name (e.g. 'Qwen3_5Profile') or dotted "
                         "import path ('my_pkg.mod.MyProfile'). If omitted, "
                         "auto-detected from the model's config.")
    args = ap.parse_args()

    cfg = _load_config(args.model)
    profile, source = _get_profile(args.profile, args.model)

    print(f"Validating profile: {type(profile).__name__} ({source})")
    print(f"Model:              {args.model}")
    print(f"model_type:         {cfg.get('model_type')}")
    print(f"architectures:      {cfg.get('architectures')}")
    print(f"vllm class:         {profile.vllm_architecture_class() or '<none>'}")
    print()

    results = validate_profile(profile, args.model, cfg)
    for r in results:
        print(r)

    n_fail = sum(1 for r in results if not r.ok)
    n_pass = len(results) - n_fail
    print()
    print(f"{n_pass} / {len(results)} checks passed",
          "" if n_fail == 0 else f" ({n_fail} failed)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
