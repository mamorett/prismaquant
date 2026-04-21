"""interaction_refine.py — sparse interaction-aware refinement near the knee.

The main allocator remains additive and cheap. This module adds a bounded
second stage:

  1. Collapse serving-tied tensors into refinement units
  2. Select the most important units near a base assignment
  3. Refine them with sparse pairwise interaction terms under the same budget

This follows the spirit of recent interaction-aware MPQ work without turning
the whole problem into a dense quadratic program over every layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .allocator import Candidate, _shape_from_stats, _group_by_profile


@dataclass(frozen=True)
class UnitOption:
    fmt: str
    bits_total: float
    predicted_dloss: float


@dataclass
class RefinementUnit:
    key: str
    members: tuple[str, ...]
    base_fmt: str
    base_member_fmts: tuple[tuple[str, str], ...]
    options: tuple[UnitOption, ...]

    @property
    def option_map(self) -> dict[str, UnitOption]:
        return {opt.fmt: opt for opt in self.options}


def _block_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3])
    leaf = parts[-1]
    if parts[3] == "self_attn" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.self_attn.{proj}"
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                if f"{prefix}.self_attn.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    if parts[3] == "mlp" and leaf in {"gate_proj", "up_proj", "down_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.mlp.{proj}"
                for proj in ("gate_proj", "up_proj", "down_proj")
                if f"{prefix}.mlp.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    return None


def _layer_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3]) + "."
    members = tuple(sorted(n for n in present if n.startswith(prefix)))
    return members if len(members) > 1 else None


_SIBLING_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".self_attn.", ("q_proj", "k_proj", "v_proj")),
    (".mlp.", ("gate_proj", "up_proj")),
    (".mlp.shared_expert.", ("gate_proj", "up_proj")),
    (".linear_attn.", ("in_proj_qkv", "in_proj_z")),
    (".linear_attn.", ("in_proj_a", "in_proj_b")),
)


def _name_pattern_siblings(name: str, present: set[str]) -> tuple[str, ...] | None:
    """Heuristic sibling detector by name pattern. Covers the serving-
    fused families (q/k/v, gate/up, in_proj_qkvz/ba) across all Qwen/
    LLaMA/DeltaNet variants we ship. Used when a profile-based classifier
    isn't available (e.g. bare-name test inputs)."""
    for parent_marker, leaves in _SIBLING_PATTERNS:
        idx = name.rfind(parent_marker)
        if idx < 0:
            continue
        parent = name[:idx + len(parent_marker)]
        leaf = name[idx + len(parent_marker):]
        if leaf not in leaves:
            continue
        members = tuple(sorted(
            f"{parent}{cand}" for cand in leaves if f"{parent}{cand}" in present
        ))
        if len(members) > 1:
            return members
    return None


def _unit_groups(names: list[str], unit_scope: str = "sibling") -> list[tuple[str, ...]]:
    present = set(names)
    # Build the sibling-key → [names] map once. `_group_by_profile` uses
    # the profile's `fused_sibling_group` classifier (derived from the
    # vLLM class's `packed_modules_mapping`), which is the same key the
    # native exporter uses for joint NVFP4 globals — so a refinement
    # unit groups exactly what vLLM fuses at serve time. We fall back
    # to the pattern detector below when the profile can't classify a
    # name (happens with bare-name test inputs, or legacy Linears that
    # don't appear in a vLLM packed_modules_mapping).
    from .model_profiles import DefaultProfile
    profile = DefaultProfile()
    sibling_key_to_names = _group_by_profile(list(present), profile)
    name_to_fusion = {
        name: tuple(sorted(members))
        for members in sibling_key_to_names.values()
        for name in members
    }

    groups: dict[tuple[str, ...], tuple[str, ...]] = {}
    for name in names:
        if ".__fused__." in name:
            key = (name,)
        else:
            key = None
            if unit_scope == "layer":
                key = _layer_group_for_name(name, present)
            if unit_scope in {"block", "hybrid"}:
                key = _block_group_for_name(name, present)
            if unit_scope == "layer" and key is None:
                key = _layer_group_for_name(name, present)
            if key is None:
                sibs = name_to_fusion.get(name)
                if sibs is not None and len(sibs) > 1:
                    key = sibs
            if key is None:
                sibs = _name_pattern_siblings(name, present)
                if sibs is not None:
                    key = sibs
            if key is None:
                key = (name,)
            else:
                key = tuple(sorted(set(key)))
        groups[key] = tuple(sorted(set(key)))
    return sorted(groups.values())


def build_refinement_units(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    assignment: dict[str, str],
    unit_scope: str = "sibling",
) -> list[RefinementUnit]:
    units = []
    for members in _unit_groups(list(assignment.keys()), unit_scope=unit_scope):
        base_fmts = {assignment[m] for m in members}
        base_member_fmts = tuple((member, assignment[member]) for member in members)
        heterogeneous_base = len(base_fmts) != 1
        base_fmt = "__base__" if heterogeneous_base else next(iter(base_fmts))
        fmt_sets = [{cand.fmt for cand in candidates[m]} for m in members if m in candidates]
        if not fmt_sets:
            continue
        shared = set.intersection(*fmt_sets)
        options = []
        if heterogeneous_base:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                cand = next(c for c in candidates[member] if c.fmt == assignment[member])
                n_params = stats[member]["n_params"]
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(fmt="__base__", bits_total=bits_total, predicted_dloss=predicted))
        for fmt in shared:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                shape = _shape_from_stats(stats[member])
                n_params = stats[member]["n_params"]
                cand = next(c for c in candidates[member] if c.fmt == fmt)
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(fmt=fmt, bits_total=bits_total, predicted_dloss=predicted))
        options.sort(key=lambda opt: (opt.bits_total, opt.predicted_dloss, opt.fmt))
        if not options:
            continue
        key = "|".join(members)
        units.append(
                RefinementUnit(
                    key=key,
                    members=members,
                    base_fmt=base_fmt,
                    base_member_fmts=base_member_fmts,
                    options=tuple(options),
                )
            )
    return units


def select_critical_units(units: list[RefinementUnit], top_n: int) -> list[RefinementUnit]:
    scored = []
    for unit in units:
        opt_map = unit.option_map
        base = opt_map[unit.base_fmt]
        cheapest = min(unit.options, key=lambda opt: (opt.bits_total, opt.predicted_dloss))
        gain = max(cheapest.predicted_dloss - base.predicted_dloss, 0.0)
        scored.append((gain, base.predicted_dloss, unit.key, unit))
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [row[-1] for row in scored[:top_n]]


def neighborhood_options(unit: RefinementUnit, radius: int = 1) -> tuple[UnitOption, ...]:
    opts = list(unit.options)
    idx = next((i for i, opt in enumerate(opts) if opt.fmt == unit.base_fmt), 0)
    lo = max(0, idx - radius)
    hi = min(len(opts), idx + radius + 1)
    return tuple(opts[lo:hi])


def base_assignment_for_units(units: list[RefinementUnit]) -> dict[str, str]:
    return {unit.key: unit.base_fmt for unit in units}


def expand_unit_assignment(units: list[RefinementUnit], choices: dict[str, str]) -> dict[str, str]:
    out = {}
    for unit in units:
        fmt = choices.get(unit.key, unit.base_fmt)
        if fmt == "__base__":
            for member, member_fmt in unit.base_member_fmts:
                out[member] = member_fmt
        else:
            for member in unit.members:
                out[member] = fmt
    return out


def objective_delta(
    choices: dict[str, str],
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
) -> float:
    total = 0.0
    for unit in units:
        total += unary.get(unit.key, {}).get(choices.get(unit.key, unit.base_fmt), 0.0)
    for left, right in combinations(sorted(choices), 2):
        lfmt = choices[left]
        rfmt = choices[right]
        total += pairwise.get((left, lfmt, right, rfmt), 0.0)
    return total


def _bits_total_for_choices(
    choices: dict[str, str],
    unit_map: dict[str, RefinementUnit],
    fixed_bits_total: float,
) -> float:
    total = fixed_bits_total
    for unit_key, fmt in choices.items():
        total += unit_map[unit_key].option_map[fmt].bits_total
    return total


def _candidate_choice_maps(units: list[RefinementUnit], allowed: dict[str, tuple[UnitOption, ...]] | None):
    out = {}
    for unit in units:
        opts = allowed[unit.key] if allowed and unit.key in allowed else unit.options
        out[unit.key] = {opt.fmt: opt for opt in opts}
    return out


def sparse_local_refine(
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
    target_total_bits: float,
    fixed_bits_total: float,
    allowed: dict[str, tuple[UnitOption, ...]] | None = None,
    max_passes: int = 8,
) -> dict:
    unit_map = {unit.key: unit for unit in units}
    option_maps = _candidate_choice_maps(units, allowed)
    current = {unit.key: unit.base_fmt for unit in units}
    current_bits = _bits_total_for_choices(current, unit_map, fixed_bits_total)
    if current_bits > target_total_bits + 1e-6:
        raise ValueError("base refinement state exceeds target budget")
    current_obj = objective_delta(current, units, unary, pairwise)

    for _pass in range(max_passes):
        best = None

        # Single-unit moves.
        for unit in units:
            for fmt in option_maps[unit.key]:
                if fmt == current[unit.key]:
                    continue
                trial = dict(current)
                trial[unit.key] = fmt
                bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                if bits > target_total_bits + 1e-6:
                    continue
                obj = objective_delta(trial, units, unary, pairwise)
                if obj + 1e-12 < current_obj and (best is None or obj < best[0]):
                    best = (obj, bits, trial)

        # Pair moves capture most of the useful interaction space while
        # remaining cheap for N≈16-32 critical units.
        for left, right in combinations(units, 2):
            for lfmt in option_maps[left.key]:
                for rfmt in option_maps[right.key]:
                    if lfmt == current[left.key] and rfmt == current[right.key]:
                        continue
                    trial = dict(current)
                    trial[left.key] = lfmt
                    trial[right.key] = rfmt
                    bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                    if bits > target_total_bits + 1e-6:
                        continue
                    obj = objective_delta(trial, units, unary, pairwise)
                    if obj + 1e-12 < current_obj and (best is None or obj < best[0]):
                        best = (obj, bits, trial)

        if best is None:
            break
        current_obj, current_bits, current = best

    return {
        "choices": current,
        "objective_delta": current_obj,
        "bits_total": current_bits,
        "bits_per_param": None,
    }


def make_pair_key(left_unit: str, left_fmt: str, right_unit: str, right_fmt: str):
    if left_unit <= right_unit:
        return (left_unit, left_fmt, right_unit, right_fmt)
    return (right_unit, right_fmt, left_unit, left_fmt)
