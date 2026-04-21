#!/usr/bin/env python3
"""streaming_model.py — shared streaming-skeleton infrastructure.

Factored out of `incremental_probe.py` so the cost-measurement side
(`incremental_measure_quant_cost.py`) can reuse the exact same
"skeleton-on-meta, head-resident, decoder-layers-swap" plumbing without
copy-pasting.

What lives here:

  - `StreamingContext`: holds the model, per-layer install resolvers,
    weight map, LayerCache, and a single-worker prefetch pool. Built once,
    reused across every shard.
  - `_build_streaming_context`: one-time setup (AutoConfig, empty
    skeleton, `from_pretrained` with explicit device_map pinning head
    resident and decoder layers to disk, strip accelerate hooks, unload
    layers back to meta).
  - `_classify_shard`: maps a shard-include regex to one of
    {"body", "mtp", "visual", "lm_head"}.

What stays in `incremental_probe`:
  - `build_layer_shard_regexes` / `build_extended_shard_regexes`
  - `load_num_hidden_layers`
  - Body/MTP shard runners (those are Fisher-semantics-specific).

The cost side will import from both this module and
`incremental_probe` (for the regex builders) — the regex helpers are
stable public API that both sides share.
"""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .layer_streaming import (
    LayerCache,
    _build_install_resolver,
    _build_weight_map,
    _fast_install,
    _get_layer_list,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
)


# ---------------------------------------------------------------------------
# Shard classification. Each shard regex falls into exactly one of these
# kinds and is orchestrated by the matching runner in the probe / cost
# script. "body" and "mtp" are the active paths; "visual" is acknowledged
# but skipped in the text-only streaming pipeline.
# ---------------------------------------------------------------------------
_BODY_SHARD_RE = re.compile(r"^model\\\.layers\\\.")
_MTP_SHARD_RE = re.compile(r"^mtp\\\.layers\\\.")
_VISUAL_SHARD_RE = re.compile(r"^model\\\.visual\\\.")
_LM_HEAD_SHARD_RE = re.compile(r"^\^lm_head\$?$")


def _classify_shard(regex: str) -> str:
    if _BODY_SHARD_RE.match(regex):
        return "body"
    if _MTP_SHARD_RE.match(regex):
        return "mtp"
    if _VISUAL_SHARD_RE.match(regex):
        return "visual"
    if _LM_HEAD_SHARD_RE.match(regex):
        return "lm_head"
    return "body"  # conservative fallback: treat as a body pattern


# ---------------------------------------------------------------------------
# Streaming context: skeleton + head resident + per-layer resolvers + cache.
# Built once for the whole run and reused across every shard. Holding this
# object idle between shards costs the head weights + cache RAM only;
# decoder layers live on meta or on disk and get installed transiently.
# ---------------------------------------------------------------------------
class StreamingContext:
    def __init__(self, *, model, base_model, layers, layers_prefix: str,
                 num_layers: int, install_resolvers: list[dict],
                 weight_shard: dict[str, str], weight_ckpt: dict[str, str],
                 layer_cache: LayerCache, prefetch_pool: ThreadPoolExecutor,
                 device: torch.device, dtype: torch.dtype, offload_folder: str):
        self.model = model
        self.base_model = base_model
        self.layers = layers
        self.layers_prefix = layers_prefix
        self.num_layers = num_layers
        self.install_resolvers = install_resolvers
        self.weight_shard = weight_shard
        self.weight_ckpt = weight_ckpt
        self.layer_cache = layer_cache
        self.prefetch_pool = prefetch_pool
        self.device = device
        self.dtype = dtype
        self.offload_folder = offload_folder
        self._inflight: dict[int, Any] = {}
        self._inflight_lock = threading.Lock()

    def _prefetch_worker(self, L: int):
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype, self.device)
        self.layer_cache.put(L, tensors)
        with self._inflight_lock:
            self._inflight.pop(L, None)
        return tensors

    def schedule_prefetch(self, L: int):
        if L < 0 or L >= self.num_layers:
            return None
        if self.layer_cache.peek(L):
            return None
        with self._inflight_lock:
            if L in self._inflight:
                return self._inflight[L]
            fut = self.prefetch_pool.submit(self._prefetch_worker, L)
            self._inflight[L] = fut
            return fut

    def ensure_loaded(self, L: int) -> tuple[dict[str, torch.Tensor], str]:
        cached = self.layer_cache.get(L)
        if cached is not None:
            return cached, "hot"
        with self._inflight_lock:
            fut = self._inflight.get(L)
        if fut is not None:
            fut.result()
            cached = self.layer_cache.get(L)
            if cached is not None:
                return cached, "wait"
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype, self.device)
        self.layer_cache.put(L, tensors)
        return tensors, "cold"

    def install(self, L: int):
        tensors, src = self.ensure_loaded(L)
        _fast_install(self.install_resolvers[L], tensors, self.device, model=self.model)
        return src

    def unload(self, L: int):
        _unload(self.model, [f"{self.layers_prefix}{L}."])

    def shutdown(self):
        self.prefetch_pool.shutdown(wait=True)


def _build_streaming_context(model_path: str, *,
                             device: torch.device, dtype: torch.dtype,
                             offload_folder: str,
                             cache_headroom_gb: float = 75.0,
                             log_prefix: str = "[streaming]") -> StreamingContext:
    """One-time setup: AutoConfig + empty skeleton, from_pretrained with an
    explicit device_map that pins the head resident and every decoder layer
    to disk, then strip accelerate's auto-load hooks and unload each layer
    back to meta so WE own materialization from this point on."""
    import psutil
    from accelerate import init_empty_weights
    from accelerate.hooks import remove_hook_from_module
    from transformers import AutoConfig, AutoModelForCausalLM

    from .sensitivity_probe import stage_text_only

    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)

    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    skel_base, skel_layers = _get_layer_list(skeleton)
    base_prefix = _resolve_base_prefix(skeleton, skel_base)
    num_layers = len(skel_layers)
    del skeleton, skel_base, skel_layers

    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    base = base_prefix if base_prefix else ""
    device_map: dict[str, object] = {}
    resident_device = 0 if device.type == "cuda" else "cpu"
    for pfx in (f"{base}.embed_tokens" if base else "embed_tokens",
                f"{base}.norm" if base else "norm",
                f"{base}.rotary_emb" if base else "rotary_emb",
                "lm_head"):
        device_map[pfx] = resident_device
    for L in range(num_layers):
        device_map[f"{base}.layers.{L}" if base else f"layers.{L}"] = "disk"

    os.makedirs(offload_folder, exist_ok=True)
    t0 = time.time()
    print(f"{log_prefix} base_prefix={base_prefix!r}  layers={num_layers}  "
          f"head_resident_on={resident_device}  offload={offload_folder}",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        staged,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
        offload_buffers=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)

    for L in range(num_layers):
        remove_hook_from_module(layers[L], recurse=True)
    for L in range(num_layers):
        _unload(model, [f"{layers_prefix}{L}."])

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    print(f"{log_prefix} model ready in {time.time()-t0:.1f}s", flush=True)

    print(f"{log_prefix} building install resolvers for {num_layers} layers ...",
          flush=True)
    t_res = time.time()
    install_resolvers = [
        _build_install_resolver(model, f"{layers_prefix}{L}".rstrip("."))
        for L in range(num_layers)
    ]
    print(f"{log_prefix} resolvers built: "
          f"{sum(len(r) for r in install_resolvers)} tensors across "
          f"{num_layers} layers in {time.time()-t_res:.1f}s", flush=True)

    free_bytes = psutil.virtual_memory().available
    cache_bytes = max(int(free_bytes) - int(cache_headroom_gb * 1024 ** 3),
                      8 * 1024 ** 3)
    layer_cache = LayerCache(max_bytes=cache_bytes)
    print(f"{log_prefix} layer cache budget={cache_bytes/(1024**3):.1f} GB "
          f"(free={free_bytes/(1024**3):.1f} GB)", flush=True)

    # Modern NVMe handles concurrent queue depth well; 3 workers saturate
    # the drive and eliminate prefetcher-stall windows where main's compute
    # drains the ahead-queue faster than a single reader can refill it.
    # Going higher (>3) sees diminishing returns and risks cache thrash.
    prefetch_pool = ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="prefetch")

    return StreamingContext(
        model=model, base_model=base_model, layers=layers,
        layers_prefix=layers_prefix, num_layers=num_layers,
        install_resolvers=install_resolvers,
        weight_shard=weight_shard, weight_ckpt=weight_ckpt,
        layer_cache=layer_cache, prefetch_pool=prefetch_pool,
        device=device, dtype=dtype, offload_folder=offload_folder,
    )
