# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import collections
from dataclasses import dataclass
import random
import json
import math
import os
from pathlib import Path
import signal
import tarfile
import threading
import time
import secrets
import sys
import traceback
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .modules import transformer as diag_transformer
from .modules.streaming import _flatten_streaming_state
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog, random_id


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False, mute_recovery_secs: float = 0.0,
                 diag_interval_secs: float = 5.0, diag_probs: bool = False, diag_dir: str = ".",
                 diag_dump_near: int = 0, diag_dump_radius: int = 50, diag_track_norms: bool = False,
                 investigate_collapse: bool = False, investigate_start: int = 5500,
                 investigate_end: int = 6500, investigate_dir: str = "runtime_logs",
                 experiment: str = "A", experiment_e_refresh_interval: int = 500,
                 experiment_f_reset_at: int = 5000):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.mute_recovery_secs = mute_recovery_secs
        self.diag_interval_secs = diag_interval_secs
        self.diag_probs = diag_probs or investigate_collapse  # investigation always needs logits
        self.diag_dir = diag_dir
        self.diag_dump_near = diag_dump_near
        self.diag_dump_radius = diag_dump_radius
        self.investigate_collapse = investigate_collapse
        self.investigate_start = investigate_start
        self.investigate_end = investigate_end
        self.investigate_dir = investigate_dir
        self.experiment = experiment
        self.experiment_e_refresh_interval = experiment_e_refresh_interval
        self.experiment_f_reset_at = experiment_f_reset_at
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
                            return_logits=self.diag_probs,
        )
        self.lm_gen.diag_track_norms = diag_track_norms or investigate_collapse
        self.lm_gen.diag_keep_hidden_tensor = investigate_collapse
        if experiment == "D":
            # Experiment D: force greedy decoding. use_sampling is an existing, already-
            # supported LMGen constructor flag -- no new mechanism needed, just flipped.
            self.lm_gen.use_sampling = False

        os.makedirs(diag_dir, exist_ok=True)
        self._diag_path = os.path.join(diag_dir, "personaplex_diag.jsonl")
        self._diag_fh = open(self._diag_path, "a", encoding="utf-8")
        self._deep_dump_path = os.path.join(diag_dir, "personaplex_deep_dump.jsonl")
        self._deep_dump_fh = open(self._deep_dump_path, "a", encoding="utf-8") if diag_dump_near > 0 else None

        if investigate_collapse:
            os.makedirs(investigate_dir, exist_ok=True)
            self._inv_dir = investigate_dir
            self._inv_fh = {
                name: open(os.path.join(investigate_dir, f"{name}.jsonl"), "a", encoding="utf-8")
                for name in ("logits", "hidden_state", "attention", "ringkvcache", "rope", "pad_events")
            }
        else:
            self._inv_dir = investigate_dir
            self._inv_fh = {}

        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Watchdog: the per-frame perf log showed steady, healthy processing right up until it
        # abruptly stops producing any output at all -- a sudden hang mid-frame, not a gradual
        # slowdown. `py-spy`/ptrace is unavailable in this container, so instead of relying on an
        # external tool or a manually-sent signal, a background thread inside this same process
        # watches a "last progress" timestamp that the hot loop below updates before every
        # individual sub-step (encode / each LM step / decode / send). If that timestamp stops
        # advancing for longer than `watchdog_timeout`, the watchdog thread (which keeps running
        # even if the main thread is blocked inside a C/CUDA call, as long as that call releases
        # the GIL while waiting) dumps every thread's current Python stack straight to the log --
        # showing exactly which named sub-step it was stuck in, automatically, the moment it hangs.
        self.watchdog_label = "idle"
        self.watchdog_ts = time.time()
        self._watchdog_reported = False

    def _mark_progress(self, label: str):
        self.watchdog_label = label
        self.watchdog_ts = time.time()
        self._watchdog_reported = False

    def _snapshot_lm_gen_state(self) -> dict:
        """Deep-clone `self.lm_gen`'s entire streaming state (every attention layer's rotating
        KV-cache plus LMGen's own token cache/offset) into plain tensors, right after the voice +
        text prompt have just finished loading. Used by mute recovery to jump straight back to
        "persona just loaded" without re-running ~1000+ full model steps one token at a time.
        """
        tensors: dict = {}
        metadata: dict = {}
        _flatten_streaming_state(tensors, metadata, self.lm_gen.get_streaming_state(), prefix="")
        snapshot = {k: v.clone() for k, v in tensors.items()}
        snapshot.update(metadata)
        return snapshot

    def _restore_lm_gen_state(self, snapshot: dict) -> None:
        """Copy a snapshot from `_snapshot_lm_gen_state` back into the live model state, in place.

        This must copy *values* into the existing tensors (via `set_streaming_state_inplace`,
        which uses `.copy_()`) rather than swap in new tensor objects: `state.graphed_main` /
        `state.graphed_depth` are CUDA-graph-captured, and a CUDA graph replay always reads and
        writes the exact memory addresses it was captured with. Replacing the tensor objects
        instead of their contents would silently desync the graph from the "restored" state.
        """
        self.lm_gen.set_streaming_state_inplace(dict(snapshot))

    def _watchdog_loop(self, timeout: float = 8.0, poll_interval: float = 2.0):
        while True:
            time.sleep(poll_interval)
            if self.watchdog_label == "idle":
                # No active session -- nothing is supposed to be making progress.
                continue
            stalled_for = time.time() - self.watchdog_ts
            if stalled_for > timeout and not self._watchdog_reported:
                self._watchdog_reported = True
                lines = [
                    "",
                    "=" * 20 + f" WATCHDOG: no progress for {stalled_for:.1f}s "
                    f"(stuck at step: {self.watchdog_label!r}) " + "=" * 20,
                ]
                for thread_id, thread_frame in sys._current_frames().items():
                    lines.append(f"thread {thread_id}:")
                    lines.extend(
                        "  " + line for line in "".join(traceback.format_stack(thread_frame)).splitlines()
                    )
                lines.append("=" * 60)
                print("\n".join(lines), file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Forensic instrumentation for the "goes silent at ~6:05, deterministically,
    # regardless of conversation content" investigation. Everything under this banner
    # is purely observational: it never changes model behavior, only records it.
    # See the final report (posted alongside this change) for what to look at first.
    # ------------------------------------------------------------------

    def _diag_write(self, record: dict) -> None:
        record.setdefault("wall_time", time.time())
        self._diag_fh.write(json.dumps(record, default=str) + "\n")
        self._diag_fh.flush()

    def _diag_event(self, conn_id: str, clog, kind: str, **fields) -> None:
        """Log a discrete, timestamped event to both the JSONL diag stream (machine-
        readable) and the human-readable server log (so it's visible without needing to
        open a second file)."""
        rec = {"record_type": "event", "conn_id": conn_id, "event": kind}
        rec.update(fields)
        self._diag_write(rec)
        summary = " ".join(f"{k}={v}" for k, v in fields.items())
        clog.log("warning", f"DIAG EVENT [{kind}] {summary}")

    def _diag_gpu_cpu_stats(self) -> dict:
        stats: dict = {}
        try:
            stats["gpu_util_pct"] = torch.cuda.utilization()
        except Exception:
            stats["gpu_util_pct"] = None
        try:
            stats["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            stats["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
        except Exception:
            stats["gpu_mem_allocated_mb"] = None
            stats["gpu_mem_reserved_mb"] = None
        try:
            stats["cpu_load_avg_1m"] = os.getloadavg()[0]
        except (AttributeError, OSError):
            stats["cpu_load_avg_1m"] = None
        return stats

    def _diag_task_summary(self) -> list:
        out = []
        try:
            tasks = asyncio.all_tasks()
        except RuntimeError:
            tasks = []
        for t in tasks:
            exc = None
            if t.done() and not t.cancelled():
                try:
                    e = t.exception()
                    exc = repr(e) if e is not None else None
                except asyncio.CancelledError:
                    exc = "cancelled"
            out.append({
                "name": t.get_name(),
                "done": t.done(),
                "cancelled": t.cancelled() if t.done() else False,
                "exception": exc,
            })
        return out

    def _diag_snapshot(self, conn_id: str, label: str, elapsed: float, extra: dict | None = None) -> dict:
        """Write a full forensic snapshot to its own JSON file (metadata only, never raw
        tensor contents) and a pointer record to the JSONL stream. Safe to call rarely
        (a handful of times per connection) -- unsafe to call every step (per-layer
        `.stats()` does a GPU sync per layer)."""
        snap = {
            "record_type": "snapshot",
            "conn_id": conn_id,
            "label": label,
            "wall_time": time.time(),
            "elapsed_session_s": round(elapsed, 2),
            "lm_gen": self.lm_gen.diagnostic_snapshot(per_layer=True, deep=True),
            "gpu_cpu": self._diag_gpu_cpu_stats(),
            "asyncio_tasks": self._diag_task_summary(),
        }
        if extra:
            snap.update(extra)
        path = os.path.join(self.diag_dir, f"snapshot_{conn_id}_{label}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        self._diag_write({
            "record_type": "snapshot_ref", "conn_id": conn_id, "label": label,
            "elapsed_session_s": round(elapsed, 2), "path": path,
        })
        return snap

    @staticmethod
    def _diag_diff(a: dict, b: dict, path: str = "") -> list:
        """Generic recursive diff between two (possibly nested) diagnostic dicts/lists of
        dicts. Returns a flat list of (dotted_path, old_value, new_value) for every leaf
        that differs. Used to automatically compare the session-start snapshot against
        the silence-onset snapshot."""
        diffs: list = []
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a.keys()) | set(b.keys())):
                p = f"{path}.{k}" if path else k
                diffs.extend(ServerState._diag_diff(a.get(k, "<missing>"), b.get(k, "<missing>"), p))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and \
                all(isinstance(x, dict) for x in a) and all(isinstance(x, dict) for x in b):
            for i, (ea, eb) in enumerate(zip(a, b)):
                diffs.extend(ServerState._diag_diff(ea, eb, f"{path}[{i}]"))
        else:
            if a != b:
                diffs.append((path, a, b))
        return diffs

    def _diag_deep_dump_step(self, conn_id: str, offset: int, elapsed: float,
                              text_logits: torch.Tensor, sampled_token: int) -> None:
        """Q3 of the mechanism investigation: full top-20 + entropy + PAD/EPAD probability
        for one generation step. Only called for steps within `diag_dump_radius` of
        `diag_dump_near`, so the cost (a sort over the ~32k-token vocabulary) is bounded
        to a small window, not paid on every step.
        """
        if self._deep_dump_fh is None:
            return
        probs = torch.softmax(text_logits.float() / self.lm_gen.temp_text, dim=-1).reshape(-1)
        top_probs, top_ids = torch.topk(probs, 20)
        top20 = [
            {"token_id": int(tid), "prob": float(p),
             "piece": self.text_tokenizer.id_to_piece(int(tid)).replace("▁", " ")}
            for tid, p in zip(top_ids.tolist(), top_probs.tolist())
        ]
        p_clamped = probs.clamp_min(1e-12)
        entropy = float(-(p_clamped * p_clamped.log()).sum().item())
        record = {
            "record_type": "deep_dump",
            "conn_id": conn_id,
            "offset": offset,
            "elapsed_s": round(elapsed, 2),
            "temperature_text": self.lm_gen.temp_text,
            "top_k_text": self.lm_gen.top_k_text,
            "use_sampling": self.lm_gen.use_sampling,
            "entropy": entropy,
            "sampled_token_id": sampled_token,
            "sampled_token_piece": self.text_tokenizer.id_to_piece(sampled_token).replace("▁", " ") if sampled_token not in (0, self.lm_gen.zero_text_code) else ("EPAD" if sampled_token == 0 else "PAD"),
            "sampled_token_prob": float(probs[sampled_token].item()),
            "pad_prob": float(probs[self.lm_gen.zero_text_code].item()),
            "epad_prob": float(probs[0].item()),
            "top20": top20,
        }
        self._deep_dump_fh.write(json.dumps(record, default=str) + "\n")
        self._deep_dump_fh.flush()

    # ------------------------------------------------------------------
    # Deep runtime investigation mode (--investigate-collapse): automatically captures
    # logits, hidden-state, attention, RingKVCache, and RoPE evidence in a narrow window
    # around state.offset, plus optional A/B experiments (--experiment A-F), so ONE
    # reproduction is enough to prove or reject each remaining hypothesis about why
    # generation permanently collapses around offset ~2x context capacity. Everything
    # here is gated to the [--investigate-start, --investigate-end] window (except
    # Experiment E/F's mid-session actions, which are deliberately session-wide) so a
    # normal, non-investigation run pays zero cost.
    # ------------------------------------------------------------------

    def _inv_write(self, stream: str, record: dict) -> None:
        fh = self._inv_fh.get(stream)
        if fh is None:
            return
        record.setdefault("wall_time", time.time())
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()

    def _inv_log_logits(self, conn_id: str, offset: int, elapsed: float,
                         text_logits: torch.Tensor, sampled_token: int) -> tuple[float, float]:
        """Q2/Q3: full top-50 distribution, entropy, PAD/EPAD probability, and whether
        the sampler actually agreed with the model's own argmax. Returns (pad_prob, entropy)
        for use by the PAD-attractor and report logic."""
        probs = torch.softmax(text_logits.float() / self.lm_gen.temp_text, dim=-1).reshape(-1)
        top_probs, top_ids = torch.topk(probs, 50)
        top50 = [
            {"token_id": int(t), "prob": float(p),
             "piece": self.text_tokenizer.id_to_piece(int(t)).replace("▁", " ")}
            for t, p in zip(top_ids.tolist(), top_probs.tolist())
        ]
        p_clamped = probs.clamp_min(1e-12)
        entropy = float(-(p_clamped * p_clamped.log()).sum().item())
        argmax_token = int(torch.argmax(probs).item())
        pad_prob = float(probs[self.lm_gen.zero_text_code].item())
        self._inv_write("logits", {
            "record_type": "logits", "conn_id": conn_id, "offset": offset, "elapsed_s": round(elapsed, 2),
            "top50": top50,
            "pad_prob": pad_prob,
            "epad_prob": float(probs[0].item()),
            "entropy": entropy,
            "argmax_token": argmax_token,
            "sampled_token": sampled_token,
            "sampled_equals_argmax": sampled_token == argmax_token,
            "temperature": self.lm_gen.temp_text,
            "top_k": self.lm_gen.top_k_text,
            "use_sampling": self.lm_gen.use_sampling,
        })
        return pad_prob, entropy

    def _inv_log_hidden(self, conn_id: str, offset: int, elapsed: float, inv_state: dict) -> Optional[float]:
        """Q7: hidden-state norm, cosine similarity + L2 distance to the previous step's
        hidden state, and a running average -- looking for a sudden collapse, saturation,
        or explosion in the single representation shared by the text head and depformer."""
        h = self.lm_gen._diag_last_hidden_state
        norm = self.lm_gen._diag_last_hidden_norm
        if h is None or norm is None:
            return None
        prev = inv_state.get("prev_hidden")
        cosine = l2 = None
        if prev is not None:
            cosine = float(torch.nn.functional.cosine_similarity(h.reshape(1, -1), prev.reshape(1, -1)).item())
            l2 = float((h - prev).norm().item())
        inv_state["hidden_running_sum"] = inv_state.get("hidden_running_sum", 0.0) + norm
        inv_state["hidden_running_count"] = inv_state.get("hidden_running_count", 0) + 1
        running_avg = inv_state["hidden_running_sum"] / inv_state["hidden_running_count"]
        self._inv_write("hidden_state", {
            "record_type": "hidden_state", "conn_id": conn_id, "offset": offset, "elapsed_s": round(elapsed, 2),
            "hidden_state_norm": norm,
            "cosine_similarity_to_prev": cosine,
            "l2_distance_to_prev": l2,
            "running_average_norm": running_avg,
        })
        inv_state["prev_hidden"] = h.clone()
        return norm

    def _inv_log_attention(self, conn_id: str, offset: int, elapsed: float) -> None:
        """Q4: drains transformer.DIAG_ATTENTION_LOG (populated by every main-transformer
        layer's forward this step -- see StreamingMultiheadAttention.forward) and clears
        it, so it never grows beyond one step's worth of entries in memory. Only
        non-empty while DIAG_CAPTURE_ATTENTION is toggled on, which this class only does
        while inside the investigation window (see _inv_step)."""
        entries = list(diag_transformer.DIAG_ATTENTION_LOG)
        diag_transformer.DIAG_ATTENTION_LOG.clear()
        if not entries:
            return
        self._inv_write("attention", {
            "record_type": "attention", "conn_id": conn_id, "offset": offset, "elapsed_s": round(elapsed, 2),
            "layers": entries,
            "entropy_mean_across_layers": sum(e["entropy_mean"] for e in entries) / len(entries),
            "max_weight_mean_across_layers": sum(e["max_weight_mean"] for e in entries) / len(entries),
            "active_kv_mean_across_layers": sum(e["active_kv_entries"] for e in entries) / len(entries),
        })

    def _inv_log_ringkvcache(self, conn_id: str, offset: int, elapsed: float) -> int:
        """Q5: per-layer write pointer / oldest / newest / wrap count / boundary-slot
        verification, for every main-transformer layer. Returns the mismatch count so the
        caller can track it for the auto-report."""
        layers = self.lm_gen.lm_model.transformer.layers
        mismatches = 0
        layer_records = []
        for i, layer in enumerate(layers):
            attn_state = layer.self_attn._streaming_state
            if attn_state is None:
                continue
            s = attn_state.kv_cache.stats()
            check = attn_state.kv_cache.verify_boundary_slot()
            if check.get("mismatch"):
                mismatches += 1
            layer_records.append({"layer": i, **s, "boundary_check": check})
        self._inv_write("ringkvcache", {
            "record_type": "ringkvcache", "conn_id": conn_id, "offset": offset, "elapsed_s": round(elapsed, 2),
            "layers": layer_records,
            "mismatch_count": mismatches,
        })
        return mismatches

    def _inv_log_rope(self, conn_id: str, offset: int, elapsed: float) -> None:
        """Q6: absolute vs. rebased (if Experiment C is active) position fed to RoPE, plus
        the resulting angle range -- computed directly from the offset and model config
        in plain Python (NOT by touching rope.py's torch.compile'd apply_rope), so this
        carries zero risk to the actual inference path. Layer 0 is representative: all
        layers share the same offset and RoPE config."""
        layer0 = self.lm_gen.lm_model.transformer.layers[0].self_attn
        dim_per_head = layer0.embed_dim // layer0.num_heads
        rope_mod = self.lm_gen.lm_model.transformer.rope
        max_period = rope_mod.max_period if rope_mod is not None else None
        rebase_context = diag_transformer.EXPERIMENT_ROPE_REBASE_CONTEXT
        rope_position = (offset % rebase_context) if rebase_context else offset
        record = {
            "record_type": "rope", "conn_id": conn_id, "offset": offset, "elapsed_s": round(elapsed, 2),
            "absolute_position": offset,
            "rope_position_used": rope_position,
            "rebase_active": rebase_context is not None,
        }
        if max_period and dim_per_head and dim_per_head > 2:
            half = dim_per_head // 2
            min_freq = math.exp(-(half - 1) * (math.log(max_period) * 2 / dim_per_head))
            record["max_angle_radians"] = rope_position * 1.0  # highest-frequency component, freq=1
            record["min_angle_radians"] = rope_position * min_freq  # lowest-frequency component
            record["max_period"] = max_period
        self._inv_write("rope", record)

    def _inv_check_pad_events(self, conn_id: str, offset: int, elapsed: float,
                               pad_prob: float, inv_state: dict) -> None:
        """Q7 (PAD attractor): fires once per threshold, the first time PAD probability
        crosses it, with the last 50 generated tokens and current hidden/attention/logits
        summary attached -- pins down exactly when PAD first becomes dominant."""
        crossed = inv_state.setdefault("pad_thresholds_crossed", set())
        for threshold in (0.10, 0.20, 0.50, 0.80, 0.95):
            if pad_prob >= threshold and threshold not in crossed:
                crossed.add(threshold)
                self._inv_write("pad_events", {
                    "record_type": "pad_event", "conn_id": conn_id, "offset": offset,
                    "elapsed_s": round(elapsed, 2), "threshold": threshold, "pad_prob": pad_prob,
                    "recent_tokens": list(inv_state.get("recent_tokens", [])),
                    "hidden_state_norm": self.lm_gen._diag_last_hidden_norm,
                })

    def _inv_save_pt_snapshot(self, label: str) -> str:
        """Full streaming-state snapshot (every layer's real K/V tensors, not just
        metadata) saved via torch.save for later offline comparison -- reuses the exact
        same snapshot format already trusted for mute-recovery (_snapshot_lm_gen_state)."""
        path = os.path.join(self._inv_dir, f"{label}.pt")
        torch.save(self._snapshot_lm_gen_state(), path)
        return path

    def _inv_generate_report(self, conn_id: str, inv_state: dict) -> str:
        """Builds investigation_report.md from what was ACTUALLY measured this run --
        every claim below is derived from inv_state's accumulated data, not asserted.
        Sections that can't be filled from this run's data say so explicitly rather than
        omitting them silently."""
        lines = ["# Investigation Report", "", f"conn_id: `{conn_id}`  experiment: `{self.experiment}`",
                  f"window: offset [{self.investigate_start}, {self.investigate_end}]", ""]

        entropies = inv_state.get("entropy_values", [])
        pad_probs = inv_state.get("pad_prob_values", [])
        hidden_norms = inv_state.get("hidden_norm_values", [])
        mismatch_total = inv_state.get("ringkv_mismatch_total", 0)
        thresholds_crossed = sorted(inv_state.get("pad_thresholds_crossed", set()))
        offsets_logged = inv_state.get("offsets_logged", [])

        lines += ["## Proven facts (directly measured this run)", ""]
        if offsets_logged:
            lines.append(f"- Dense logging captured {len(offsets_logged)} steps, "
                          f"offset {offsets_logged[0]} to {offsets_logged[-1]}.")
        if pad_probs:
            lines.append(f"- PAD probability ranged from {min(pad_probs):.4f} to {max(pad_probs):.4f} "
                         f"across the window; final value {pad_probs[-1]:.4f}.")
        if thresholds_crossed:
            lines.append(f"- PAD probability crossed thresholds {thresholds_crossed} during this run "
                         f"(see pad_events.jsonl for the exact offset/context of each).")
        else:
            lines.append("- PAD probability never crossed the 10% threshold in this window -- "
                         "either the collapse didn't happen in this run/window, or it happens "
                         "outside [--investigate-start, --investigate-end] for this configuration.")
        if hidden_norms:
            change_pct = 100.0 * (hidden_norms[-1] - hidden_norms[0]) / max(abs(hidden_norms[0]), 1e-6)
            lines.append(f"- Hidden-state norm: start={hidden_norms[0]:.3f}, end={hidden_norms[-1]:.3f} "
                         f"({change_pct:+.1f}% change over the window).")
        lines.append(f"- RingKVCache boundary-slot mismatches observed: {mismatch_total} "
                     f"(Experiment B active: {self.experiment == 'B'}).")

        lines += ["", "## Rejected hypotheses (this run's evidence)", ""]
        if mismatch_total == 0 and self.experiment == "B":
            lines.append("- RingKVCache boundary off-by-one: absent in this run (Experiment B fix active) "
                         "-- if collapse still occurred, that rules out this bug as the sole cause.")
        if hidden_norms and abs(100.0 * (hidden_norms[-1] - hidden_norms[0]) / max(abs(hidden_norms[0]), 1e-6)) < 5:
            lines.append("- Hidden-state exploding/vanishing: NOT supported -- norm changed by "
                         "less than 5% across the window.")

        lines += ["", "## Supported hypotheses (this run's evidence)", ""]
        if entropies and len(entropies) > 1 and entropies[-1] < 0.5 * entropies[0]:
            lines.append(f"- Attention/output collapse toward a narrow distribution: entropy dropped "
                         f"from {entropies[0]:.3f} to {entropies[-1]:.3f} ({100*(1-entropies[-1]/entropies[0]):.0f}% "
                         f"reduction) across the window -- consistent with a collapse into a low-entropy attractor.")
        if hidden_norms and len(hidden_norms) > 1 and abs(100.0 * (hidden_norms[-1] - hidden_norms[0]) / max(abs(hidden_norms[0]), 1e-6)) >= 5:
            lines.append(f"- Hidden-state drift: norm changed by "
                         f"{100.0 * (hidden_norms[-1] - hidden_norms[0]) / max(abs(hidden_norms[0]), 1e-6):+.1f}% "
                         f"across the window -- worth comparing against Experiment C (RoPE rebase) to see "
                         f"if this drift shrinks.")
        if mismatch_total > 0 and self.experiment == "A":
            lines.append(f"- RingKVCache boundary mismatch is real and active in this run "
                         f"({mismatch_total} occurrences logged) -- compare against an Experiment B "
                         f"run to see if fixing it changes the collapse offset.")

        lines += ["", "## Remaining unknowns", ""]
        lines.append("- Whether this specific run's pattern (gradual ramp vs. sudden jump in PAD "
                     "probability) generalizes across different personas/voice prompts -- this run "
                     "only characterizes one configuration.")
        if self.experiment == "A":
            lines.append("- Whether Experiments B-F individually shift, delay, or eliminate the collapse "
                         "-- requires running each and comparing its own investigation_report.md against "
                         "this baseline.")
        lines.append("- The model's actual trained maximum position / sequence length is not documented "
                     "in this repository and cannot be determined from runtime evidence alone.")

        lines += ["", "## Suggested next experiment", ""]
        if self.experiment == "A":
            if pad_probs and thresholds_crossed and 0.95 in thresholds_crossed:
                # Find how abrupt the final approach to collapse was.
                lines.append("- Run **Experiment C** (`--experiment C`, RoPE rebase) with the same "
                             "--investigate-start/--investigate-end: if PAD probability no longer "
                             "reaches the thresholds it did here, absolute RoPE position is implicated; "
                             "if the pattern is unchanged, RoPE can be ruled out with high confidence.")
            else:
                lines.append("- Widen the window (`--investigate-start`/`--investigate-end`) and rerun "
                             "baseline: this run's window may not have bracketed the actual collapse point.")
        else:
            lines.append(f"- Compare this Experiment {self.experiment} run's investigation_report.md "
                         f"against the Experiment A baseline's: did the offset of first PAD-threshold "
                         f"crossing move, and by how much?")

        report = "\n".join(lines) + "\n"
        # Named investigation_report_<conn>_<experiment>.md rather than the literal
        # investigation_report.md -- running multiple experiments (the whole point of the
        # A-F comparison) into the same shared directory would otherwise have each run
        # silently overwrite the previous one's report. Same reasoning for
        # experiment_summary.json below.
        report_path = os.path.join(self._inv_dir, f"investigation_report_{conn_id}_{self.experiment}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        summary = {
            "conn_id": conn_id,
            "experiment": self.experiment,
            "window": [self.investigate_start, self.investigate_end],
            "steps_logged": len(offsets_logged),
            "offset_range_logged": [offsets_logged[0], offsets_logged[-1]] if offsets_logged else None,
            "pad_prob_range": [min(pad_probs), max(pad_probs)] if pad_probs else None,
            "pad_prob_final": pad_probs[-1] if pad_probs else None,
            "pad_thresholds_crossed": thresholds_crossed,
            "entropy_range": [min(entropies), max(entropies)] if entropies else None,
            "entropy_final": entropies[-1] if entropies else None,
            "hidden_norm_range": [min(hidden_norms), max(hidden_norms)] if hidden_norms else None,
            "hidden_norm_start_end": [hidden_norms[0], hidden_norms[-1]] if hidden_norms else None,
            "ringkv_mismatch_total": mismatch_total,
            "experiment_b_fix_active": self.experiment == "B",
            "experiment_c_rope_rebase_active": self.experiment == "C",
            "window_ever_reached": bool(offsets_logged),
            "report_path": report_path,
        }
        summary_path = os.path.join(self._inv_dir, f"experiment_summary_{conn_id}_{self.experiment}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return report_path

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                # When --diag-probs is on, LMGen was constructed with return_logits=True, so
                # step() always returns a (tokens, logits) 2-tuple instead of just tokens --
                # including during this warmup, before any connection exists.
                if self.diag_probs:
                    tokens, _logits = self.lm_gen.step(codes[:, :, c: c + 1])
                else:
                    tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_chat(self, request):
        # `heartbeat` makes aiohttp send a WS ping every N seconds and expect a pong back
        # within N/2 seconds (the browser answers pings automatically, invisibly to JS).
        # Without this, a long-lived connection that momentarily has no application data to
        # send (e.g. a quiet stretch where the Opus encoder emits little/no bytes) looks
        # "idle" to any intermediary (reverse proxy, load balancer, NAT/firewall) sitting
        # between the browser and this server, several of which silently drop connections
        # after a few minutes of perceived inactivity. The periodic ping keeps the path warm
        # and also lets aiohttp itself detect and close a truly dead peer quickly instead of
        # a task hanging forever on a half-open socket.
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        diag_conn_id = random_id()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")
        clog.log("info", f"diag_conn_id={diag_conn_id} diag_file={self._diag_path}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path
                
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        seed = int(request["seed"]) if "seed" in request.query else None

        # Shared, mutable diagnostic counters -- a plain dict works across the three
        # coroutines below without needing `nonlocal` for each field (only reassigning
        # the name `diag_state` itself would need that; mutating its contents doesn't).
        diag_state = {
            "bytes_received": 0,
            "bytes_sent_audio": 0,
            "bytes_sent_text": 0,
            "pcm_frames_received": 0,
            "mimi_encode_calls": 0,
            "encoded_code_frames": 0,
            "pcm_frames_decoded": 0,
            "ws_last_recv_time": time.time(),
            "ws_last_send_time": time.time(),
        }

        # --investigate-collapse state, declared here (not inside opus_loop) so it's
        # still readable after the session ends -- see the fallback report generation
        # at the end of this function, for sessions that close before state.offset ever
        # reaches --investigate-end.
        inv_state = {
            "recent_tokens": collections.deque(maxlen=50),
            "entropy_values": [], "pad_prob_values": [], "hidden_norm_values": [],
            "offsets_logged": [], "ringkv_mismatch_total": 0,
            "entered": False, "finished": False,
        }

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                        diag_state["bytes_received"] += len(payload)
                        diag_state["ws_last_recv_time"] = time.time()
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            except Exception as exc:
                self._diag_event(diag_conn_id, clog, "exception", where="recv_loop",
                                  exception=repr(exc), traceback=traceback.format_exc())
                raise
            finally:
                close = True
                self._diag_event(diag_conn_id, clog, "websocket_disconnect",
                                  where="recv_loop", ws_closed=ws.closed)
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None
            frame_wall_time = self.frame_size / self.mimi.sample_rate
            frames_since_report = 0
            processing_time_since_report = 0.0
            encode_time_since_report = 0.0
            lm_step_time_since_report = 0.0
            decode_time_since_report = 0.0
            last_report = time.time()
            last_text_time = time.time()
            text_tokens_since_report = 0
            pad_tokens_since_report = 0
            epad_tokens_since_report = 0
            out_sq_sum = 0.0
            out_sample_cnt = 0
            out_peak_since_report = 0.0
            recovery_count = 0

            # --- forensic instrumentation state (see _diag_* helpers on ServerState) ---
            capacity = self.lm_gen.lm_model.context
            session_conv_start = time.time()
            prev_wrap_count = 0
            prev_milestone = 0
            capacity_reached_logged = False
            capacity_exceeded_logged = False
            silence_event_fired = False
            frame_number = 0
            silent_frame_count = 0
            consecutive_silent_frames = 0
            SILENCE_RMS_THRESHOLD = 0.001
            prob_sampled_sum = 0.0
            prob_pad_sum = 0.0
            prob_count = 0
            # Time-based snapshot schedule: (elapsed_seconds, label). Popped in order as
            # `session_conv_start`-relative elapsed time crosses each threshold.
            pending_snapshots = [
                (120, "t120s"), (180, "t180s"), (240, "t240s"), (300, "t300s"),
                (330, "t330s"), (345, "t345s"), (360, "t360s"),
            ]

            # --- --investigate-collapse state (see the _inv_* methods on ServerState) ---
            # `inv_state` itself is declared in handle_chat's scope (not here), so it's
            # still readable after this coroutine ends, for the fallback report generation.
            inv_e_last_refresh_offset = 0
            inv_f_reset_done = False

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                now0 = time.time()
                elapsed = now0 - session_conv_start
                while pending_snapshots and elapsed >= pending_snapshots[0][0]:
                    _, label = pending_snapshots.pop(0)
                    self._diag_snapshot(diag_conn_id, label, elapsed)
                    clog.log("info", f"DIAG snapshot '{label}' taken at {elapsed:.1f}s elapsed")

                if self.investigate_collapse:
                    _cur_off = self.lm_gen._streaming_state.offset
                    # Experiment E: periodic persona refresh -- re-feed the text prompt
                    # tokens through step() as-is (no state reset), exactly like initial
                    # prompt loading, so they naturally occupy new ring slots. Session-wide
                    # by design (testing whether this prevents the collapse from ever
                    # being reached, not just observing it inside the narrow window).
                    if self.experiment == "E" and self.lm_gen.text_prompt_tokens and \
                            _cur_off - inv_e_last_refresh_offset >= self.experiment_e_refresh_interval:
                        inv_e_last_refresh_offset = _cur_off
                        self._mark_progress("opus_loop: experiment E persona refresh")
                        refresh_start = time.time()
                        self.lm_gen._step_text_prompt()
                        self._diag_event(diag_conn_id, clog, "experiment_e_refresh",
                                          offset_before=_cur_off, duration_s=round(time.time() - refresh_start, 2))
                    # Experiment F: one-time hidden-state reset at a configured offset,
                    # WITHOUT reconnecting -- reuses the exact same instant snapshot/restore
                    # already trusted for mute recovery, just triggered by offset instead of
                    # by silence, and only once, to test whether resetting before the
                    # collapse point prevents it.
                    if self.experiment == "F" and not inv_f_reset_done and _cur_off >= self.experiment_f_reset_at:
                        inv_f_reset_done = True
                        self._mark_progress("opus_loop: experiment F one-time reset")
                        self._restore_lm_gen_state(persona_snapshot)
                        _ = opus_reader.read_pcm()
                        all_pcm_data = None
                        self._diag_event(diag_conn_id, clog, "experiment_f_reset", offset=_cur_off)

                if self.mute_recovery_secs > 0 and (time.time() - last_text_time) > self.mute_recovery_secs:
                    # The model has not produced a single word in `mute_recovery_secs`. In our
                    # observed failure mode this state is permanent (the persona/system prompt has
                    # rotated out of the model's fixed 3000-step attention window and generation
                    # degenerates into endless PAD tokens), so rather than stay silent forever we
                    # jump the model's entire streaming state (every attention layer's KV-cache,
                    # LMGen's own token cache/offset) back to the snapshot taken right after the
                    # voice+text prompt finished loading at connection time. This is a handful of
                    # in-place tensor copies, not ~1000+ replayed forward passes, so it takes
                    # milliseconds instead of tens of seconds -- no audio backlog builds up and no
                    # mic input is dropped while it happens.
                    recovery_count += 1
                    reprime_start = time.time()
                    self._mark_progress("opus_loop: mute recovery (instant persona restore)")
                    self._diag_event(diag_conn_id, clog, "recovery_triggered",
                                      recovery_count=recovery_count, elapsed_s=round(elapsed, 1),
                                      offset=self.lm_gen._streaming_state.offset)
                    self._restore_lm_gen_state(persona_snapshot)
                    # The just-restored state predates anything the user said since connecting, so
                    # audio queued up under the old (dead) state must be dropped -- feeding it into
                    # the freshly-restored state would splice unrelated context together.
                    _ = opus_reader.read_pcm()
                    all_pcm_data = None
                    last_text_time = time.time()
                    clog.log(
                        "warning",
                        f"model was silent for >{self.mute_recovery_secs:.0f}s -- restored persona "
                        f"in {time.time() - reprime_start:.3f}s (recovery #{recovery_count}). "
                        f"Note: this restores the persona, not the conversation since then -- the "
                        f"model's fixed attention window cannot hold both.",
                    )
                    continue

                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                diag_state["pcm_frames_received"] += pcm.shape[-1]
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    try:
                        be = time.time()
                        frame_number += 1
                        self._mark_progress("opus_loop: slicing/transfer chunk")
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]

                        t_enc = time.time()
                        self._mark_progress("opus_loop: mimi.encode")
                        codes = self.mimi.encode(chunk)
                        self._mark_progress("opus_loop: other_mimi.encode")
                        _ = self.other_mimi.encode(chunk)
                        encode_time_since_report += time.time() - t_enc
                        diag_state["mimi_encode_calls"] += 1
                        diag_state["encoded_code_frames"] += codes.shape[-1]

                        for c in range(codes.shape[-1]):
                            current_offset = self.lm_gen._streaming_state.offset
                            self._mark_progress(
                                f"opus_loop: lm_gen.step (c={c}/{codes.shape[-1]}, offset={current_offset})"
                            )

                            # --- cheap, per-step, sync-free offset/wrap tracking ---
                            # `current_offset` is a plain Python int already materialized by
                            # LMGen (no GPU read needed here). Every main-transformer layer's
                            # own RingKVCache advances in lockstep with it (each is fed exactly
                            # one token per step()), so wrap/position arithmetic on this single
                            # counter is equivalent to reading the real cache, without the sync
                            # cost of calling `.stats()` (which does) on every step.
                            wrap_count = current_offset // capacity
                            if wrap_count != prev_wrap_count:
                                self._diag_event(diag_conn_id, clog, "ring_cache_wrap",
                                                  wrap_count=wrap_count, offset=current_offset,
                                                  elapsed_s=round(elapsed, 1))
                                prev_wrap_count = wrap_count
                            milestone = current_offset // 250
                            if milestone != prev_milestone:
                                self._diag_event(diag_conn_id, clog, "offset_milestone",
                                                  offset=current_offset, milestone=milestone * 250,
                                                  elapsed_s=round(elapsed, 1))
                                prev_milestone = milestone
                            if current_offset == capacity and not capacity_reached_logged:
                                capacity_reached_logged = True
                                self._diag_event(diag_conn_id, clog, "offset_reached_capacity",
                                                  offset=current_offset, capacity=capacity,
                                                  elapsed_s=round(elapsed, 1))
                            if current_offset > capacity and not capacity_exceeded_logged:
                                capacity_exceeded_logged = True
                                self._diag_event(diag_conn_id, clog, "offset_exceeded_capacity",
                                                  offset=current_offset, capacity=capacity,
                                                  elapsed_s=round(elapsed, 1))

                            t_lm = time.time()
                            if self.diag_probs:
                                tokens, logits_pack = self.lm_gen.step(codes[:, :, c: c + 1])
                                text_logits = logits_pack[0] if logits_pack is not None else None
                            else:
                                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                                text_logits = None
                            lm_step_time_since_report += time.time() - t_lm
                            if tokens is None:
                                continue
                            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

                            t_dec = time.time()
                            self._mark_progress("opus_loop: mimi.decode")
                            main_pcm = self.mimi.decode(tokens[:, 1:9])
                            self._mark_progress("opus_loop: other_mimi.decode")
                            _ = self.other_mimi.decode(tokens[:, 1:9])
                            decode_time_since_report += time.time() - t_dec
                            diag_state["pcm_frames_decoded"] += 1

                            self._mark_progress("opus_loop: main_pcm.cpu()")
                            main_pcm = main_pcm.cpu()
                            self._mark_progress("opus_loop: opus_writer.append_pcm")
                            pcm_out = main_pcm[0, 0].numpy()
                            opus_writer.append_pcm(pcm_out)
                            frame_rms = float(np.sqrt(np.mean(np.square(pcm_out))))
                            frame_peak = float(np.abs(pcm_out).max()) if pcm_out.size else 0.0
                            out_sq_sum += float(np.square(pcm_out).sum())
                            out_sample_cnt += pcm_out.shape[-1]
                            out_peak_since_report = max(out_peak_since_report, frame_peak)
                            if frame_rms < SILENCE_RMS_THRESHOLD:
                                silent_frame_count += 1
                                consecutive_silent_frames += 1
                                if consecutive_silent_frames == 1:
                                    self._diag_event(diag_conn_id, clog, "output_rms_near_zero",
                                                      frame_rms=round(frame_rms, 6), frame_number=frame_number,
                                                      elapsed_s=round(elapsed, 1))
                            else:
                                consecutive_silent_frames = 0

                            text_token = tokens[0, 0, 0].item()
                            if text_logits is not None:
                                # Divide by temp_text before softmax to match the actual
                                # distribution `sample_token()` samples from internally
                                # (see moshi/moshi/utils/sampling.py) -- otherwise these
                                # numbers would be a different (temp=1) distribution than
                                # the one that actually produced the sampled token.
                                probs = torch.softmax(text_logits.float() / self.lm_gen.temp_text, dim=-1)
                                prob_sampled_sum += float(probs[0, 0, 0, text_token].item())
                                prob_pad_sum += float(probs[0, 0, 0, self.lm_gen.zero_text_code].item())
                                prob_count += 1
                                if self.diag_dump_near > 0 and abs(current_offset - self.diag_dump_near) <= self.diag_dump_radius:
                                    self._diag_deep_dump_step(diag_conn_id, current_offset, elapsed, text_logits, text_token)

                            if self.investigate_collapse:
                                in_window = self.investigate_start <= current_offset <= self.investigate_end
                                if in_window and not inv_state["entered"]:
                                    inv_state["entered"] = True
                                    diag_transformer.DIAG_CAPTURE_ATTENTION = True
                                    snap_path = self._inv_save_pt_snapshot("before_silence_snapshot")
                                    self._diag_event(diag_conn_id, clog, "investigation_window_entered",
                                                      offset=current_offset, snapshot=snap_path)
                                if in_window and not inv_state["finished"]:
                                    inv_state["offsets_logged"].append(current_offset)
                                    if text_logits is not None:
                                        pad_prob, entropy = self._inv_log_logits(
                                            diag_conn_id, current_offset, elapsed, text_logits, text_token)
                                        inv_state["pad_prob_values"].append(pad_prob)
                                        inv_state["entropy_values"].append(entropy)
                                        self._inv_check_pad_events(diag_conn_id, current_offset, elapsed, pad_prob, inv_state)
                                    hnorm = self._inv_log_hidden(diag_conn_id, current_offset, elapsed, inv_state)
                                    if hnorm is not None:
                                        inv_state["hidden_norm_values"].append(hnorm)
                                    self._inv_log_attention(diag_conn_id, current_offset, elapsed)
                                    inv_state["ringkv_mismatch_total"] += self._inv_log_ringkvcache(
                                        diag_conn_id, current_offset, elapsed)
                                    self._inv_log_rope(diag_conn_id, current_offset, elapsed)
                                    piece = ("PAD" if text_token == self.lm_gen.zero_text_code
                                              else "EPAD" if text_token == 0
                                              else self.text_tokenizer.id_to_piece(text_token).replace("▁", " "))
                                    inv_state["recent_tokens"].append(piece)
                                elif current_offset > self.investigate_end and not inv_state["finished"]:
                                    inv_state["finished"] = True
                                    diag_transformer.DIAG_CAPTURE_ATTENTION = False
                                    after_path = self._inv_save_pt_snapshot("after_silence_snapshot")
                                    report_path = self._inv_generate_report(diag_conn_id, inv_state)
                                    self._diag_event(diag_conn_id, clog, "investigation_window_finished",
                                                      offset=current_offset, snapshot=after_path, report=report_path)
                                    clog.log("warning", f"INVESTIGATION COMPLETE: report written to {report_path}")

                            if text_token == 0:
                                epad_tokens_since_report += 1
                            elif text_token == self.lm_gen.zero_text_code:  # 3 == PAD
                                pad_tokens_since_report += 1
                            else:
                                text_tokens_since_report += 1
                                last_text_time = time.time()
                                silence_event_fired = False
                                _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                                _text = _text.replace("▁", " ")
                                msg = b"\x02" + bytes(_text, encoding="utf8")
                                self._mark_progress("opus_loop: ws.send_bytes (text)")
                                await ws.send_bytes(msg)
                                diag_state["bytes_sent_text"] += len(msg)
                                diag_state["ws_last_send_time"] = time.time()

                            if not silence_event_fired and (time.time() - last_text_time) > 5.0:
                                silence_event_fired = True
                                silence_elapsed = time.time() - session_conv_start
                                self._diag_event(diag_conn_id, clog, "silence_onset",
                                                  elapsed_s=round(silence_elapsed, 1),
                                                  offset=current_offset)
                                silence_snap = self._diag_snapshot(
                                    diag_conn_id, "silence_onset", silence_elapsed,
                                )
                                diffs = self._diag_diff(
                                    session_start_snapshot.get("lm_gen", {}),
                                    silence_snap.get("lm_gen", {}),
                                )
                                clog.log(
                                    "warning",
                                    f"DIAG: {len(diffs)} field(s) changed between session_start and "
                                    f"silence_onset lm_gen snapshots (see {self._diag_path} for full detail)",
                                )
                                self._diag_write({
                                    "record_type": "auto_diff",
                                    "conn_id": diag_conn_id,
                                    "compared": ["session_start", "silence_onset"],
                                    "diff_count": len(diffs),
                                    "diffs": [{"path": p, "before": a, "after": b} for p, a, b in diffs],
                                })
                                pending_snapshots.append(
                                    (silence_elapsed + 30, "silence_plus_30s")
                                )
                                pending_snapshots.sort()

                        processing_time_since_report += time.time() - be
                        frames_since_report += 1
                        self._mark_progress("opus_loop: between frames")
                    except Exception as exc:
                        self._diag_event(diag_conn_id, clog, "exception", where="opus_loop",
                                          exception=repr(exc), traceback=traceback.format_exc())
                        raise

                now = time.time()
                if self.diag_interval_secs > 0 and now - last_report >= self.diag_interval_secs:
                    backlog_s = (0 if all_pcm_data is None else all_pcm_data.shape[-1]) / self.mimi.sample_rate
                    budget_s = frames_since_report * frame_wall_time
                    rtf = (processing_time_since_report / budget_s) if budget_s > 0 else 0.0
                    out_rms = (out_sq_sum / out_sample_cnt) ** 0.5 if out_sample_cnt > 0 else 0.0
                    current_offset = self.lm_gen._streaming_state.offset
                    clog.log(
                        "info" if rtf < 0.9 else "warning",
                        f"perf: {frames_since_report} frames in last {now - last_report:.1f}s, "
                        f"processing/real-time ratio={rtf:.2f}, unprocessed input backlog={backlog_s:.2f}s, "
                        f"text_tokens={text_tokens_since_report}, out_rms={out_rms:.4f}, "
                        f"last_text={now - last_text_time:.0f}s ago, "
                        f"offset={current_offset}/{capacity}",
                    )
                    diag_record = {
                        "record_type": "periodic",
                        "conn_id": diag_conn_id,
                        "time": {
                            "elapsed_session_s": round(now - session_conv_start, 2),
                            "frame_number": frame_number,
                            "model_step_count": current_offset,
                            "state_offset": current_offset,
                            "ring_wrap_count": current_offset // capacity,
                            "modulo_index": current_offset % capacity,
                            "context_capacity": capacity,
                            "pct_context_consumed": round(100.0 * current_offset / capacity, 2),
                        },
                        "lm_state": {
                            "lm_gen_offset": current_offset,
                            "streaming_state_summary": self.lm_gen.diagnostic_snapshot(per_layer=False),
                            "text_tokens_count": text_tokens_since_report,
                            "pad_tokens_count": pad_tokens_since_report,
                            "epad_tokens_count": epad_tokens_since_report,
                            "avg_prob_sampled_token": (prob_sampled_sum / prob_count) if prob_count else None,
                            "avg_prob_pad": (prob_pad_sum / prob_count) if prob_count else None,
                            "diag_probs_enabled": self.diag_probs,
                        },
                        "audio_pipeline": {
                            "pcm_frames_received_total": diag_state["pcm_frames_received"],
                            "mimi_encode_calls_total": diag_state["mimi_encode_calls"],
                            "encoded_code_frames_total": diag_state["encoded_code_frames"],
                            "mic_queue_backlog_s": round(backlog_s, 3),
                            "pcm_frames_decoded_total": diag_state["pcm_frames_decoded"],
                            "output_rms": round(out_rms, 6),
                            "output_peak": round(out_peak_since_report, 6),
                            "silent_frame_count_total": silent_frame_count,
                            "consecutive_silent_frames": consecutive_silent_frames,
                        },
                        "connection": {
                            "ws_closed": ws.closed,
                            "seconds_since_last_recv": round(now - diag_state["ws_last_recv_time"], 2),
                            "seconds_since_last_send": round(now - diag_state["ws_last_send_time"], 2),
                            "asyncio_tasks": self._diag_task_summary(),
                        },
                        "performance": {
                            **self._diag_gpu_cpu_stats(),
                            "mimi_encode_time_s": round(encode_time_since_report, 4),
                            "lm_step_time_s": round(lm_step_time_since_report, 4),
                            "mimi_decode_time_s": round(decode_time_since_report, 4),
                            "processing_real_time_ratio": round(rtf, 3),
                        },
                    }
                    self._diag_write(diag_record)
                    last_report = now
                    frames_since_report = 0
                    processing_time_since_report = 0.0
                    encode_time_since_report = 0.0
                    lm_step_time_since_report = 0.0
                    decode_time_since_report = 0.0
                    text_tokens_since_report = 0
                    pad_tokens_since_report = 0
                    epad_tokens_since_report = 0
                    out_sq_sum = 0.0
                    out_sample_cnt = 0
                    out_peak_since_report = 0.0
                    prob_sampled_sum = 0.0
                    prob_pad_sum = 0.0
                    prob_count = 0

        async def send_loop():
            try:
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)
                    self._mark_progress("send_loop: opus_writer.read_bytes")
                    msg = opus_writer.read_bytes()
                    if len(msg) > 0:
                        self._mark_progress("send_loop: ws.send_bytes (audio)")
                        await ws.send_bytes(b"\x01" + msg)
                        diag_state["bytes_sent_audio"] += len(msg)
                        diag_state["ws_last_send_time"] = time.time()
            except Exception as exc:
                self._diag_event(diag_conn_id, clog, "exception", where="send_loop",
                                  exception=repr(exc), traceback=traceback.format_exc())
                raise

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Snapshot right after priming so mid-conversation mute recovery (see opus_loop) can
            # jump back here almost instantly instead of replaying the whole prompt.
            persona_snapshot = self._snapshot_lm_gen_state()
            # Full forensic snapshot at the same point, used as the "known good" baseline that
            # the silence-onset snapshot (taken inside opus_loop) gets automatically diffed
            # against the moment the model goes quiet.
            session_start_snapshot = self._diag_snapshot(diag_conn_id, "session_start", elapsed=0.0)
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop(), name="recv_loop"),
                    asyncio.create_task(opus_loop(), name="opus_loop"),
                    asyncio.create_task(send_loop(), name="send_loop"),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Surface any exception that ended the session early instead of letting it
                # vanish silently (an un-retrieved task exception is otherwise only ever
                # logged much later, if at all, when the task object is garbage collected).
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        clog.log("error", f"session task {task.get_name()} failed: {exc!r}")
                    self._diag_event(diag_conn_id, clog, "task_exit", task=task.get_name(),
                                      exception=repr(exc) if exc is not None else None)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        if self.investigate_collapse and not inv_state["finished"]:
            # The session ended (disconnect, error, or the user just stopped talking)
            # before state.offset ever reached --investigate-end. Generate the report
            # anyway from whatever was captured, rather than silently producing nothing --
            # it will say plainly if the window was never reached or only partially covered.
            inv_state["finished"] = True
            diag_transformer.DIAG_CAPTURE_ATTENTION = False
            if inv_state["entered"]:
                after_path = self._inv_save_pt_snapshot("after_silence_snapshot")
            else:
                after_path = None
                self._diag_event(diag_conn_id, clog, "investigation_window_never_reached",
                                  final_offset=self.lm_gen._streaming_state.offset if self.lm_gen.is_streaming else None)
            report_path = self._inv_generate_report(diag_conn_id, inv_state)
            clog.log("warning", f"INVESTIGATION (session ended before window closed): "
                                 f"report written to {report_path}")
        self._mark_progress("idle")
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _dump_stacks(signum, frame):
    """SIGUSR1 handler: print every asyncio task's current stack plus every Python thread's
    stack, without killing the process. Useful in containers (like some RunPod images) that
    deny the ptrace permission py-spy needs (`Permission denied (os error 13)`).

    Usage: `kill -USR1 <server_pid>`, then check the server's stdout/log file.

    Caveat: like any pure-Python signal handler, this only runs once control returns to the
    interpreter between bytecode instructions -- if the process is stuck inside a single
    long-running C/CUDA call holding the GIL, the dump won't print until that call returns.
    """
    lines = ["", "=" * 20 + " STACK DUMP (SIGUSR1) " + "=" * 20]
    try:
        tasks = asyncio.all_tasks()
    except RuntimeError:
        tasks = []
        lines.append("no running asyncio event loop found in this thread")
    for task in tasks:
        lines.append(f"--- asyncio task {task.get_name()} ---")
        stack = task.get_stack()
        if not stack:
            lines.append("  (no Python stack available -- likely awaiting a Future/executor)")
        else:
            lines.extend("  " + line for line in "".join(traceback.format_stack(stack[-1])).splitlines())
    lines.append("--- all Python threads (sys._current_frames) ---")
    for thread_id, thread_frame in sys._current_frames().items():
        lines.append(f"thread {thread_id}:")
        lines.extend("  " + line for line in "".join(traceback.format_stack(thread_frame)).splitlines())
    lines.append("=" * 60)
    print("\n".join(lines), file=sys.stderr, flush=True)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    if hasattr(signal, "SIGUSR1"):  # not available on Windows
        signal.signal(signal.SIGUSR1, _dump_stacks)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--mute-recovery-secs", type=float, default=0.0,
                        help="If > 0 and the model produces no text for this many seconds "
                             "mid-conversation (the 'goes permanently silent after a few minutes' "
                             "failure mode), automatically reset and replay the voice/text prompts "
                             "inside the same connection so the conversation can continue. "
                             "0 disables recovery (default). For a clean evidence-gathering run of "
                             "the silence issue itself, leave this at 0 so the natural failure "
                             "isn't auto-recovered away before its full signature is captured.")
    parser.add_argument("--diag-interval-secs", type=float, default=5.0,
                        help="Interval, in seconds, for the structured forensic diagnostic log "
                             "(JSONL, one line per interval covering timing, LM state, KV-cache "
                             "position, audio pipeline, connection, and performance). 0 disables "
                             "periodic diagnostic logging (event/snapshot logging still runs). "
                             "Default 5.0s.")
    parser.add_argument("--diag-probs", action="store_true",
                        help="Also capture the sampling probability assigned to the sampled text "
                             "token and to the PAD token, every step. Requires computing an extra "
                             "softmax and cloning logits every step (small but real per-step "
                             "overhead) -- off by default, turn on for one detailed diagnostic run.")
    parser.add_argument("--diag-dir", type=str, default=".",
                        help="Directory for diagnostic output: personaplex_diag.jsonl (structured "
                             "periodic/event/snapshot-pointer log) plus one snapshot_<conn>_<label>.json "
                             "file per snapshot. Defaults to the current directory.")
    parser.add_argument("--diag-dump-near", type=int, default=0,
                        help="Mechanism investigation (Q3): if > 0, write a full top-20-token / "
                             "entropy / PAD-probability dump to personaplex_diag_deep_dump.jsonl for "
                             "every step within --diag-dump-radius of this state.offset value. "
                             "Requires --diag-probs. 0 disables (default).")
    parser.add_argument("--diag-dump-radius", type=int, default=50,
                        help="Window (in steps) around --diag-dump-near to deep-dump. Default 50.")
    parser.add_argument("--diag-track-norms", action="store_true",
                        help="Mechanism investigation (Q7): track the L2 norm of the shared hidden "
                             "state (transformer_out) every step, included in snapshots. Small added "
                             "per-step cost -- off by default.")
    parser.add_argument("--investigate-collapse", action="store_true",
                        help="Deep runtime investigation mode: automatically captures logits, "
                             "hidden-state, attention, RingKVCache, and RoPE evidence when "
                             "state.offset enters [--investigate-start, --investigate-end], plus an "
                             "auto-generated investigation_report.md at the end. Forces "
                             "NO_CUDA_GRAPH=1 (required for attention capture) and --diag-probs -- "
                             "expect reduced throughput for the duration of this run.")
    parser.add_argument("--investigate-start", type=int, default=5500,
                        help="state.offset at which dense investigation logging begins. Default 5500.")
    parser.add_argument("--investigate-end", type=int, default=6500,
                        help="state.offset at which dense investigation logging ends and the "
                             "investigation_report.md is generated. Default 6500.")
    parser.add_argument("--investigate-dir", type=str, default="runtime_logs",
                        help="Directory for investigation output: logits.jsonl, hidden_state.jsonl, "
                             "attention.jsonl, ringkvcache.jsonl, rope.jsonl, pad_events.jsonl, "
                             "before/after_silence_snapshot.pt, and investigation_report_*.md. "
                             "Default './runtime_logs'.")
    parser.add_argument("--experiment", type=str, default="A", choices=["A", "B", "C", "D", "E", "F"],
                        help="A=baseline, no modification (default). B=fix the RingKVCache boundary "
                             "off-by-one (delta<0 instead of delta<=0). C=RoPE rebase (offset %% "
                             "context) for the rotation angle only, not the causal mask or cache "
                             "indexing. D=force greedy decoding (argmax, no sampling). E=periodic "
                             "persona refresh every --experiment-e-refresh-interval steps. F=one-time "
                             "LM hidden-state reset at --experiment-f-reset-at, without reconnecting.")
    parser.add_argument("--experiment-e-refresh-interval", type=int, default=500,
                        help="Experiment E: steps between persona-refresh reinjections. Default 500.")
    parser.add_argument("--experiment-f-reset-at", type=int, default=5000,
                        help="Experiment F: state.offset at which the one-time reset fires. Default 5000.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()

    if args.investigate_collapse:
        # Must happen before any model is loaded/warmed up: CUDAGraphed captures its
        # kernel sequence on first real use, and NO_CUDA_GRAPH / the experiment globals
        # below need to be in their final state before that capture happens, or the
        # capture would bake in the WRONG (pre-flag) behavior and ignore later changes.
        os.environ["NO_CUDA_GRAPH"] = "1"
        diag_transformer.DIAG_CAPTURE_ATTENTION = False  # toggled on/off around the window at runtime
        if args.experiment == "B":
            diag_transformer.EXPERIMENT_FIX_BOUNDARY_OFFBYONE = True
        if args.experiment == "C":
            diag_transformer.EXPERIMENT_ROPE_REBASE_CONTEXT = loaders._lm_kwargs["context"]
        logger.warning(
            f"--investigate-collapse: forcing NO_CUDA_GRAPH=1, running experiment "
            f"'{args.experiment}'. Expect reduced throughput for this run."
        )

    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        mute_recovery_secs=args.mute_recovery_secs,
        diag_interval_secs=args.diag_interval_secs,
        diag_probs=args.diag_probs,
        diag_dir=args.diag_dir,
        diag_dump_near=args.diag_dump_near,
        diag_dump_radius=args.diag_dump_radius,
        diag_track_norms=args.diag_track_norms,
        investigate_collapse=args.investigate_collapse,
        investigate_start=args.investigate_start,
        investigate_end=args.investigate_end,
        investigate_dir=args.investigate_dir,
        experiment=args.experiment,
        experiment_e_refresh_interval=args.experiment_e_refresh_interval,
        experiment_f_reset_at=args.experiment_f_reset_at,
    )
    if args.diag_dump_near > 0 and not args.diag_probs:
        logger.warning("--diag-dump-near requires --diag-probs to actually capture logits; "
                        "the deep-dump file will stay empty without it.")
    logger.info(f"diagnostic log: {state._diag_path}")
    if state._deep_dump_fh is not None:
        logger.info(f"deep-dump log (near offset {args.diag_dump_near}): {state._deep_dump_path}")
    if args.investigate_collapse:
        logger.info(f"investigation mode ON: experiment={args.experiment}, "
                    f"window=[{args.investigate_start}, {args.investigate_end}], "
                    f"output dir={args.investigate_dir}")
    logger.info("warming up the model")
    state.warmup()
    threading.Thread(target=state._watchdog_loop, daemon=True, name="hang-watchdog").start()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
