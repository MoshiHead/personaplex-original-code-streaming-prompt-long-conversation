# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Permanent "black box" runtime recorder for PersonaPlex live streaming sessions.

Design (see the discussion that produced this file): a flight data recorder does not
capture every sensor at full fidelity continuously -- it loops a small, cheap parameter
set forever, and only freezes the expensive detail around an actual incident. This module
is the same split, applied to one LMGen streaming session:

  Tier 1 (always on, every step, CUDA-graph-safe): cheap scalars only -- offset, PAD/EPAD
  probability, entropy, hidden-state norm and its cosine-similarity/L2-distance to the
  previous step, RingKVCache wrap/fullness, audio RMS, latency, GPU memory. None of this
  requires NO_CUDA_GRAPH; all of it is written to stream.jsonl immediately, every step, for
  the entire session.

  Tier 2 (rolling, in-memory only, bounded): a fixed-size deque of the more expensive
  per-step data (top-100 tokens, a detached hidden-state tensor, per-layer RingKVCache
  stats). This is never written to disk on its own -- it is only frozen and persisted when
  a Tier 1 metric trips one of the automatic anomaly detectors below, exactly like a
  cockpit voice recorder's rolling loop being preserved after an incident instead of
  overwritten.

  Deep, per-layer attention/activation capture (real per-layer entropy, per-head
  statistics, layer/residual outputs) is NOT part of either tier by default: reading it
  requires NO_CUDA_GRAPH=1 (see transformer.py's DIAG_CAPTURE_ATTENTION) because a
  CUDA-graph-replayed layer's Python-level forward never re-runs during replay, so there is
  no hook-based way to inspect it without disabling graphs and accepting materially lower
  throughput. `enable_attention_recording=True` opts into this trade-off explicitly; it is
  off by default so a production/live deployment is not silently slowed down.
"""
from __future__ import annotations

import collections
import json
import math
import os
import time
import wave
from typing import Optional

import numpy as np
import torch

from .modules import transformer as diag_transformer


def next_session_dir(root: str) -> tuple[str, str]:
    """Scans `root` for existing session_NNN directories and returns (session_id,
    full_path) for the next one, zero-padded to 3 digits. Scanning disk (rather than
    keeping an in-process counter) means session numbering survives server restarts."""
    os.makedirs(root, exist_ok=True)
    nums = []
    for name in os.listdir(root):
        if name.startswith("session_") and name[len("session_"):].isdigit():
            nums.append(int(name[len("session_"):]))
    n = (max(nums) + 1) if nums else 1
    session_id = f"session_{n:03d}"
    return session_id, os.path.join(root, session_id)


class RollingZScore:
    """Causal (backward-looking only) rolling mean/std tracker for one scalar metric.
    Used by the online anomaly detectors below -- same rolling-window z-score approach
    already validated against real recorded sessions (the offset~6000 change-point
    analysis), just computed incrementally as each step arrives instead of after the
    fact on a saved log."""

    def __init__(self, window: int = 30):
        self.window = window
        self.values: collections.deque = collections.deque(maxlen=window)

    def update(self, x: float) -> dict:
        mean = std = z = deriv = None
        if self.values:
            mean = sum(self.values) / len(self.values)
            deriv = x - self.values[-1]
        if len(self.values) >= 5:
            var = sum((v - mean) ** 2 for v in self.values) / len(self.values)
            std = math.sqrt(var)
            if std > 1e-9:
                z = (x - mean) / std
        self.values.append(x)
        return {"mean": mean, "std": std, "z": z, "deriv": deriv}


class SessionRecorder:
    """One instance per live connection. Construct at the start of `handle_chat`, call
    `record_step(...)` once per LM step from `opus_loop`, and call `close(...)` when the
    connection ends. Everything under `session_dir` is this session's complete black box:
    stream.jsonl (Tier 1, always), events.jsonl + snapshots/ (Tier 2, only if triggered),
    input.wav/output.wav (for literal listen-back), timeline.json, and session_report.md
    (both generated automatically in `close()`).
    """

    # Cheap, generous defaults -- see the module docstring for why these particular
    # thresholds don't need per-deployment tuning to be useful: they're set loose enough
    # to only fire on genuinely unusual steps, not everyday variance.
    PAD_LOCK_PROB_THRESHOLD = 0.9
    PAD_LOCK_STEP_COUNT = 50
    REPEATED_TOKEN_COUNT = 20
    HIDDEN_NORM_Z_THRESHOLD = 4.0
    COSINE_SIM_DROP_THRESHOLD = 0.5
    PAD_PROB_Z_THRESHOLD = 4.0
    ENTROPY_Z_THRESHOLD = 4.0
    SLOW_STEP_MS_THRESHOLD = 500.0
    LONG_SILENCE_SECS = 8.0
    ROLLING_BUFFER_STEPS = 300
    POST_EVENT_STEPS = 100

    def __init__(
        self,
        session_root: str,
        lm_gen,
        text_tokenizer,
        sample_rate: int,
        conn_id: str,
        voice_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        enable_attention_recording: bool = False,
        logger=None,
    ):
        self.session_id, self.session_dir = next_session_dir(session_root)
        os.makedirs(os.path.join(self.session_dir, "snapshots"), exist_ok=True)
        self.lm_gen = lm_gen
        self.text_tokenizer = text_tokenizer
        self.sample_rate = sample_rate
        self.conn_id = conn_id
        self.enable_attention_recording = enable_attention_recording
        self.logger = logger

        self._stream_fh = open(os.path.join(self.session_dir, "stream.jsonl"), "a", encoding="utf-8")
        self._events_fh = open(os.path.join(self.session_dir, "events.jsonl"), "a", encoding="utf-8")
        self._conversation_fh = open(os.path.join(self.session_dir, "conversation.jsonl"), "a", encoding="utf-8")

        self.session_start = time.time()
        self.turn_index = 0
        self.chunk_index = 0
        self.last_text_time = self.session_start
        self.in_silence = False

        self._rolling = {
            "pad_prob": RollingZScore(),
            "entropy": RollingZScore(),
            "hidden_norm": RollingZScore(),
            "attention_entropy": RollingZScore(),
        }
        self._prev_hidden_state: Optional[torch.Tensor] = None
        self._recent_tokens: collections.deque = collections.deque(maxlen=self.REPEATED_TOKEN_COUNT)
        self._pad_streak = 0
        self._prev_offset_wrap = 0

        # Tier 2: bounded, in-memory only until an event freezes it.
        self._rolling_buffer: collections.deque = collections.deque(maxlen=self.ROLLING_BUFFER_STEPS)
        self._active_events: dict = {}  # name -> steps remaining to also capture after trigger
        self.events: list = []
        self.generated_text_parts: list = []
        self.timeline: list = [{"offset": 0, "wall_time": self.session_start, "label": "session started"}]

        self._input_wav: Optional[wave.Wave_write] = None
        self._output_wav: Optional[wave.Wave_write] = None
        try:
            self._input_wav = self._open_wav(os.path.join(self.session_dir, "input.wav"))
            self._output_wav = self._open_wav(os.path.join(self.session_dir, "output.wav"))
        except Exception as e:
            self._log(f"session_recorder: could not open wav files for replay audio: {e!r}")

        meta = {
            "session_id": self.session_id,
            "conn_id": conn_id,
            "start_wall_time": self.session_start,
            "voice_prompt": voice_prompt,
            "text_prompt": text_prompt,
            "sample_rate": sample_rate,
            "enable_attention_recording": enable_attention_recording,
        }
        with open(os.path.join(self.session_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        self._log(f"session_recorder: {self.session_id} started -> {self.session_dir}")

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.log("info", msg)

    def _open_wav(self, path: str) -> wave.Wave_write:
        w = wave.open(path, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(self.sample_rate)
        return w

    @staticmethod
    def _pcm_float_to_int16_bytes(pcm: np.ndarray) -> bytes:
        clipped = np.clip(pcm, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    def record_input_pcm(self, pcm: np.ndarray) -> None:
        if self._input_wav is not None:
            try:
                self._input_wav.writeframes(self._pcm_float_to_int16_bytes(pcm))
            except Exception:
                pass

    def record_output_pcm(self, pcm: np.ndarray) -> None:
        if self._output_wav is not None:
            try:
                self._output_wav.writeframes(self._pcm_float_to_int16_bytes(pcm))
            except Exception:
                pass

    def record_text_token(self, piece: str) -> None:
        """Called for every non-PAD/EPAD generated text piece -- builds the full
        conversation transcript and detects turn boundaries via silence gaps."""
        now = time.time()
        if self.in_silence:
            self.turn_index += 1
            self.in_silence = False
        self.last_text_time = now
        self.generated_text_parts.append(piece)

    def _write_stream_record(self, record: dict) -> None:
        record.setdefault("wall_time", time.time())
        self._stream_fh.write(json.dumps(record, default=str) + "\n")
        self._stream_fh.flush()

    def _write_event(self, name: str, offset: int, reason: str, confidence: float,
                      metrics_before: dict, extra: Optional[dict] = None) -> dict:
        event = {
            "event": name,
            "offset": offset,
            "wall_time": time.time(),
            "elapsed_s": round(time.time() - self.session_start, 2),
            "reason": reason,
            "confidence": round(confidence, 3),
            "metrics_before": metrics_before,
            "metrics_after": None,  # filled in once the post-event window completes
            "recent_tokens": list(self._recent_tokens),
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        self.timeline.append({"offset": offset, "wall_time": event["wall_time"], "label": f"{name}: {reason}"})
        self._events_fh.write(json.dumps(event, default=str) + "\n")
        self._events_fh.flush()
        self._log(f"session_recorder EVENT [{name}] offset={offset} reason={reason} confidence={confidence:.2f}")
        return event

    def _freeze_snapshot(self, name: str, offset: int, snapshot_fn) -> Optional[str]:
        """Dumps the rolling Tier-2 buffer plus a full state snapshot to
        session_dir/snapshots/. `snapshot_fn` is ServerState._snapshot_lm_gen_state,
        passed in rather than imported to avoid a server.py <-> session_recorder.py
        import cycle."""
        try:
            path = os.path.join(self.session_dir, "snapshots", f"event_{name}_{offset}.pt")
            torch.save(snapshot_fn(), path)
            buf_path = os.path.join(self.session_dir, "snapshots", f"event_{name}_{offset}_buffer.json")
            with open(buf_path, "w", encoding="utf-8") as f:
                json.dump(list(self._rolling_buffer), f, indent=2, default=str)
            return path
        except Exception as e:
            self._log(f"session_recorder: snapshot for event {name}@{offset} failed: {e!r}")
            return None

    def record_step(
        self,
        offset: int,
        elapsed_s: float,
        text_logits: Optional[torch.Tensor],
        sampled_text_token: int,
        hidden_state: Optional[torch.Tensor],
        latency_ms: float,
        audio_rms: float,
        ringkv_capacity: int,
        snapshot_fn=None,
    ) -> list:
        """Called once per LM step from opus_loop. Returns the list of event names (if
        any) that fired this step, so the caller can log/react without re-deriving them.
        `snapshot_fn` should be `ServerState._snapshot_lm_gen_state` -- passed through
        rather than imported, deliberately, to keep this module free of a server.py
        import (session_recorder.py is meant to be reusable outside the live server,
        e.g. from run_investigation_suite or a future tool, without pulling in aiohttp)."""
        fired: list = []
        pad_prob = entropy = None
        argmax_token = None
        if text_logits is not None:
            probs = torch.softmax(text_logits.float() / self.lm_gen.temp_text, dim=-1)
            pad_prob = float(probs[0, 0, 0, self.lm_gen.zero_text_code].item())
            epad_prob = float(probs[0, 0, 0, 0].item())
            p = probs[0, 0, 0].clamp_min(1e-12)
            entropy = float(-(p * p.log()).sum().item())
            argmax_token = int(probs[0, 0, 0].argmax().item())
        else:
            epad_prob = None

        hidden_norm = cosine_sim = l2_dist = None
        if hidden_state is not None:
            hidden_norm = float(hidden_state.float().norm().item())
            if self._prev_hidden_state is not None:
                a, b = hidden_state.float().flatten(), self._prev_hidden_state.flatten()
                cosine_sim = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
                l2_dist = float((a - b).norm().item())
            self._prev_hidden_state = hidden_state.float().detach().clone()

        wrap_count = offset // ringkv_capacity if ringkv_capacity else 0

        record = {
            "record_type": "step", "session_id": self.session_id, "conn_id": self.conn_id,
            "offset": offset, "elapsed_s": round(elapsed_s, 3), "turn": self.turn_index,
            "chunk_index": self.chunk_index, "pad_prob": pad_prob, "epad_prob": epad_prob,
            "entropy": entropy, "sampled_token": sampled_text_token, "argmax_token": argmax_token,
            "hidden_norm": hidden_norm, "cosine_sim_to_prev": cosine_sim, "l2_dist_to_prev": l2_dist,
            "ringkv_wrap_count": wrap_count, "audio_rms": audio_rms, "latency_ms": round(latency_ms, 2),
        }
        try:
            record["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        except Exception:
            record["gpu_mem_allocated_mb"] = None
        self._write_stream_record(record)
        self.chunk_index += 1

        # Tier 2 rolling buffer: cheap enough to always compute (a single topk over an
        # already-materialized logits tensor is negligible next to the model's own
        # forward pass), but never written to disk unless a detector below fires.
        top_tokens = None
        if text_logits is not None:
            topk = torch.topk(probs[0, 0, 0], k=min(100, probs.shape[-1]))
            top_tokens = [
                {"token_id": int(i), "prob": float(p),
                 "piece": self.text_tokenizer.id_to_piece(int(i)).replace("▁", " ")}
                for p, i in zip(topk.values.tolist(), topk.indices.tolist())
            ]
        attention_summary = None
        if self.enable_attention_recording:
            # DIAG_CAPTURE_ATTENTION/DIAG_ATTENTION_LOG are process-wide globals in
            # transformer.py, shared by the main transformer, the depformer, and both
            # mimi instances -- drained and filtered exactly like
            # ServerState._inv_log_attention (see that method's comment for the full
            # explanation): the main transformer's own internally-read offset trails
            # this step's `offset` by exactly 1, confirmed empirically, so filtering on
            # `offset - 1` isolates just the main transformer's real per-step entries.
            # Always drained when this flag is on, every step, regardless of whether an
            # event fires -- DIAG_ATTENTION_LOG would otherwise grow without bound.
            entries = list(diag_transformer.DIAG_ATTENTION_LOG)
            diag_transformer.DIAG_ATTENTION_LOG.clear()
            main_entries = [e for e in entries if e["offset"] == offset - 1]
            if main_entries:
                attention_summary = {
                    "entropy_mean": sum(e["entropy_mean"] for e in main_entries) / len(main_entries),
                    "max_weight_mean": sum(e["max_weight_mean"] for e in main_entries) / len(main_entries),
                    "active_kv_mean": sum(e["active_kv_entries"] for e in main_entries) / len(main_entries),
                    "per_layer": main_entries,
                }

        self._rolling_buffer.append({"offset": offset, "elapsed_s": elapsed_s, **record,
                                      "top_tokens": top_tokens, "attention": attention_summary})

        # --- automatic anomaly detectors (Tier 1 scalars only -- no extra GPU work) ---
        if pad_prob is not None:
            self._pad_streak = self._pad_streak + 1 if pad_prob >= self.PAD_LOCK_PROB_THRESHOLD else 0
            if self._pad_streak == self.PAD_LOCK_STEP_COUNT:
                fired.append(self._trigger("pad_lock", offset,
                                            f"PAD probability >= {self.PAD_LOCK_PROB_THRESHOLD} for "
                                            f"{self.PAD_LOCK_STEP_COUNT} consecutive steps",
                                            confidence=1.0, metrics_before={"pad_prob": pad_prob}, snapshot_fn=snapshot_fn))
            z = self._rolling["pad_prob"].update(pad_prob)
            if z["z"] is not None and z["z"] >= self.PAD_PROB_Z_THRESHOLD:
                fired.append(self._trigger("pad_probability_rising", offset,
                                            f"PAD probability z-score {z['z']:.2f} >= {self.PAD_PROB_Z_THRESHOLD}",
                                            confidence=min(1.0, z["z"] / (2 * self.PAD_PROB_Z_THRESHOLD)),
                                            metrics_before=z, snapshot_fn=snapshot_fn))

        if entropy is not None:
            z = self._rolling["entropy"].update(entropy)
            if z["z"] is not None and abs(z["z"]) >= self.ENTROPY_Z_THRESHOLD:
                fired.append(self._trigger("abnormal_entropy", offset,
                                            f"entropy z-score {z['z']:.2f}", confidence=min(1.0, abs(z["z"]) / (2 * self.ENTROPY_Z_THRESHOLD)),
                                            metrics_before=z, snapshot_fn=snapshot_fn))

        if hidden_norm is not None:
            z = self._rolling["hidden_norm"].update(hidden_norm)
            if z["z"] is not None and abs(z["z"]) >= self.HIDDEN_NORM_Z_THRESHOLD:
                name = "activation_explosion" if z["z"] > 0 else "activation_vanishing"
                fired.append(self._trigger(name, offset, f"hidden-state norm z-score {z['z']:.2f}",
                                            confidence=min(1.0, abs(z["z"]) / (2 * self.HIDDEN_NORM_Z_THRESHOLD)),
                                            metrics_before=z, snapshot_fn=snapshot_fn))
        if cosine_sim is not None and cosine_sim < self.COSINE_SIM_DROP_THRESHOLD:
            fired.append(self._trigger("hidden_state_drift", offset,
                                        f"cosine similarity to previous step {cosine_sim:.3f} < {self.COSINE_SIM_DROP_THRESHOLD}",
                                        confidence=min(1.0, (self.COSINE_SIM_DROP_THRESHOLD - cosine_sim) / self.COSINE_SIM_DROP_THRESHOLD),
                                        metrics_before={"cosine_sim_to_prev": cosine_sim}, snapshot_fn=snapshot_fn))

        if attention_summary is not None:
            z = self._rolling["attention_entropy"].update(attention_summary["entropy_mean"])
            if z["z"] is not None and abs(z["z"]) >= self.ENTROPY_Z_THRESHOLD:
                fired.append(self._trigger("attention_collapse", offset,
                                            f"attention-entropy z-score {z['z']:.2f}",
                                            confidence=min(1.0, abs(z["z"]) / (2 * self.ENTROPY_Z_THRESHOLD)),
                                            metrics_before=z, snapshot_fn=snapshot_fn))

        if sampled_text_token is not None:
            self._recent_tokens.append(sampled_text_token)
            if len(self._recent_tokens) == self.REPEATED_TOKEN_COUNT and len(set(self._recent_tokens)) == 1:
                fired.append(self._trigger("output_loop", offset,
                                            f"same token repeated {self.REPEATED_TOKEN_COUNT} times consecutively",
                                            confidence=1.0, metrics_before={"token": sampled_text_token}, snapshot_fn=snapshot_fn))

        if wrap_count != self._prev_offset_wrap:
            self._prev_offset_wrap = wrap_count
            self._write_event("ringkv_wrap", offset, f"RingKVCache wrapped (wrap_count={wrap_count})",
                               confidence=1.0, metrics_before={"wrap_count": wrap_count})
            self.timeline.append({"offset": offset, "wall_time": time.time(),
                                   "label": f"RingKVCache wrapped (#{wrap_count})"})

        if latency_ms >= self.SLOW_STEP_MS_THRESHOLD:
            fired.append(self._trigger("slow_inference", offset, f"step latency {latency_ms:.0f}ms",
                                        confidence=min(1.0, latency_ms / (2 * self.SLOW_STEP_MS_THRESHOLD)),
                                        metrics_before={"latency_ms": latency_ms}, snapshot_fn=snapshot_fn))

        if (time.time() - self.last_text_time) >= self.LONG_SILENCE_SECS and not self.in_silence:
            self.in_silence = True
            fired.append(self._trigger("long_silence", offset,
                                        f"no text token for >= {self.LONG_SILENCE_SECS}s",
                                        confidence=1.0,
                                        metrics_before={"secs_since_last_text": time.time() - self.last_text_time},
                                        snapshot_fn=snapshot_fn))

        # Post-event follow-up: once POST_EVENT_STEPS have passed since a trigger, fill
        # in metrics_after and, for events with a real snapshot, freeze one more buffer
        # dump so the "after" state is also on disk, not just "before".
        done = []
        for name, remaining in self._active_events.items():
            if remaining <= 1:
                done.append(name)
                for e in reversed(self.events):
                    if e["event"] == name and e["metrics_after"] is None:
                        e["metrics_after"] = {"pad_prob": pad_prob, "entropy": entropy, "hidden_norm": hidden_norm}
                        break
            else:
                self._active_events[name] = remaining - 1
        for name in done:
            del self._active_events[name]

        return [f for f in fired if f]  # drop the Nones from _trigger()'s cooldown suppression

    def _trigger(self, name: str, offset: int, reason: str, confidence: float,
                 metrics_before: dict, snapshot_fn=None) -> Optional[str]:
        if name in self._active_events:
            # Already recorded once for this same ongoing anomaly -- extend its cooldown
            # instead of writing another near-duplicate event + full snapshot every single
            # step for as long as a sustained drift keeps the detector above threshold.
            # Without this, one real incident that lasts N steps would produce N events and
            # N snapshots instead of one, which defeats the point of a black box (capture
            # the incident, don't flood the log with it).
            self._active_events[name] = self.POST_EVENT_STEPS
            return None
        snapshot_path = self._freeze_snapshot(name, offset, snapshot_fn) if snapshot_fn is not None else None
        self._write_event(name, offset, reason, confidence, metrics_before,
                           extra={"snapshot_path": snapshot_path})
        self._active_events[name] = self.POST_EVENT_STEPS
        return name

    def record_exception(self, offset: int, exc: BaseException) -> None:
        self._write_event("server_exception", offset, f"{type(exc).__name__}: {exc}", confidence=1.0,
                           metrics_before={})

    def record_disconnect(self, offset: int, reason: str = "websocket closed") -> None:
        self._write_event("websocket_disconnect", offset, reason, confidence=1.0, metrics_before={})

    # ------------------------------------------------------------------
    def close(self) -> str:
        """Finalizes the session: closes all file handles, writes timeline.json, and
        generates session_report.md. Returns the report path."""
        self.timeline.append({"offset": None, "wall_time": time.time(), "label": "session ended"})
        for fh in (self._stream_fh, self._events_fh, self._conversation_fh):
            try:
                fh.close()
            except Exception:
                pass
        for w in (self._input_wav, self._output_wav):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass

        transcript = "".join(self.generated_text_parts)
        with open(os.path.join(self.session_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

        with open(os.path.join(self.session_dir, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump(self.timeline, f, indent=2, default=str)

        report_path = self._generate_report(transcript)
        return report_path

    def _generate_report(self, transcript: str) -> str:
        lines = ["# Session Report", "", f"session: `{self.session_id}`  conn_id: `{self.conn_id}`",
                  f"duration: {round(time.time() - self.session_start, 1)}s  "
                  f"turns: {self.turn_index + 1}  events: {len(self.events)}", ""]

        lines += ["## What happened", ""]
        if not self.events:
            lines.append("No automatic anomaly was detected during this session -- generation "
                         "stayed within normal bounds on every metric this recorder tracks.")
        else:
            first = self.events[0]
            lines.append(f"The first detected anomaly was **{first['event']}** at offset "
                         f"{first['offset']} ({first['elapsed_s']}s in): {first['reason']}.")
            lines.append(f"{len(self.events)} total event(s) were recorded; see the timeline below "
                         f"for the full sequence.")

        lines += ["", "## Timeline", ""]
        for point in self.timeline:
            off = point["offset"] if point["offset"] is not None else "--"
            lines.append(f"- offset {off}: {point['label']}")

        lines += ["", "## Which component/metric changed first", ""]
        if self.events:
            by_metric = {}
            for e in self.events:
                by_metric.setdefault(e["event"], e["offset"])
            ordered = sorted(by_metric.items(), key=lambda kv: kv[1])
            for name, off in ordered:
                lines.append(f"- `{name}` first fired at offset {off}")
            lines.append("")
            lines.append(f"Earliest: **`{ordered[0][0]}`** at offset {ordered[0][1]} -- treat this as "
                         f"the leading indicator, not necessarily the root cause; correlation between "
                         f"events recorded in the same session is not causation (see the causal-probe "
                         f"tooling in server.py for actually testing that).")
        else:
            lines.append("Not applicable -- no anomaly was recorded.")

        lines += ["", "## Evidence for / against a root cause", ""]
        lines.append("This report only states what this recorder directly measured. It does not "
                     "assert a mechanism beyond the ordering above -- pairing this session against a "
                     "healthy one (see the notebook's session-comparison view) and, if a real "
                     "intervention is needed, `ServerState.run_causal_probe()`, are the next steps "
                     "for anything beyond ordering and correlation.")

        lines += ["", "## Components that stayed healthy", ""]
        all_possible = {"pad_lock", "pad_probability_rising", "abnormal_entropy", "activation_explosion",
                         "activation_vanishing", "hidden_state_drift", "output_loop", "slow_inference",
                         "long_silence"}
        if self.enable_attention_recording:
            all_possible.add("attention_collapse")
        fired_names = {e["event"] for e in self.events}
        healthy = sorted(all_possible - fired_names)
        if healthy:
            for h in healthy:
                lines.append(f"- `{h}`: never triggered this session")
        else:
            lines.append("- none -- every tracked detector fired at least once this session")

        lines += ["", "## Files in this session", ""]
        lines.append("- `stream.jsonl` -- every step's Tier 1 scalars")
        lines.append("- `events.jsonl` -- every detected anomaly, in order")
        lines.append("- `snapshots/` -- full state + rolling-buffer dump for each event, if any fired")
        lines.append("- `input.wav` / `output.wav` -- the session's audio, for literal listen-back")
        lines.append("- `transcript.txt` -- the model's generated text for the whole session")
        lines.append("- `timeline.json` -- the machine-readable version of the timeline above")

        report = "\n".join(lines) + "\n"
        report_path = os.path.join(self.session_dir, "session_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        self._log(f"session_recorder: {self.session_id} closed, report -> {report_path}")
        return report_path
