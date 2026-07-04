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
from dataclasses import dataclass
import random
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
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


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
                 save_voice_prompt_embeddings: bool = False, mute_recovery_secs: float = 0.0):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.mute_recovery_secs = mute_recovery_secs
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
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

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
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
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

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
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None
            # Real-time-factor tracking: each `frame_size` chunk represents a fixed slice of
            # wall-clock audio time. If processing one chunk takes longer than that, we are
            # falling behind real time and the backlog in `all_pcm_data` will only ever grow --
            # from the client's perspective this looks exactly like a "frozen" conversation
            # (the connection stays open and audio keeps trickling out, just increasingly late)
            # even though the GPU is still busy grinding through a queue of stale input.
            frame_wall_time = self.frame_size / self.mimi.sample_rate
            frames_since_report = 0
            processing_time_since_report = 0.0
            last_report = time.time()
            # Content-level tracking: the mechanical pipeline can be perfectly healthy while the
            # model itself has gone "mute" (generating only PAD text tokens and silent audio,
            # forever). Track when the model last said anything and how loud its output is so the
            # perf log distinguishes "pipeline broken" from "model went silent".
            last_text_time = time.time()
            text_tokens_since_report = 0
            out_sq_sum = 0.0
            out_sample_cnt = 0
            recovery_count = 0

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                if self.mute_recovery_secs > 0 and (time.time() - last_text_time) > self.mute_recovery_secs:
                    # The model has not produced a single word in `mute_recovery_secs`. In our
                    # observed failure mode this state is permanent (the persona/system prompt has
                    # rotated out of the model's fixed 3000-step attention window and generation
                    # degenerates into endless PAD tokens), so rather than stay silent forever we
                    # re-prime the persona *inside the same connection*: reset the streaming
                    # states and replay the voice + text prompts, exactly like connection setup.
                    recovery_count += 1
                    clog.log(
                        "warning",
                        f"model produced no text for {time.time() - last_text_time:.0f}s -- "
                        f"re-priming persona in-session (recovery #{recovery_count}, "
                        f"takes roughly as long as the initial connect pause)",
                    )
                    reprime_start = time.time()
                    last_keepalive = 0.0

                    async def alive_midsession():
                        # Unlike the connection-setup `is_alive`, this must NOT call ws.receive()
                        # (recv_loop is running concurrently and aiohttp forbids two concurrent
                        # receives). It yields to the event loop so recv_loop/send_loop/heartbeat
                        # keep running, and sends an in-band ping (0x06, understood and ignored by
                        # the web client) every few seconds so the client's inactivity watchdog
                        # doesn't kill the connection during this deliberately-quiet stretch.
                        nonlocal last_keepalive
                        if close or ws.closed:
                            return False
                        # Refresh the hang watchdog too: a long re-prime is deliberate work,
                        # not a stall, and shouldn't trigger stack dumps.
                        self._mark_progress("opus_loop: mute recovery (re-priming persona)")
                        await asyncio.sleep(0.001)
                        if time.time() - last_keepalive > 5.0:
                            last_keepalive = time.time()
                            await ws.send_bytes(b"\x06")
                        return True

                    self._mark_progress("opus_loop: mute recovery (re-priming persona)")
                    self.mimi.reset_streaming()
                    self.other_mimi.reset_streaming()
                    self.lm_gen.reset_streaming()
                    await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=alive_midsession)
                    self.mimi.reset_streaming()
                    # Drop mic audio that piled up while the model was being re-primed (it was
                    # "deaf" during that window; feeding it stale audio would only confuse it).
                    _ = opus_reader.read_pcm()
                    all_pcm_data = None
                    last_text_time = time.time()
                    clog.log(
                        "warning",
                        f"persona re-primed in {time.time() - reprime_start:.1f}s; conversation resumes",
                    )
                    continue

                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    self._mark_progress("opus_loop: slicing/transfer chunk")
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    self._mark_progress("opus_loop: mimi.encode")
                    codes = self.mimi.encode(chunk)
                    self._mark_progress("opus_loop: other_mimi.encode")
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        self._mark_progress(f"opus_loop: lm_gen.step (c={c}/{codes.shape[-1]}, offset={self.lm_gen._streaming_state.offset})")
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        self._mark_progress("opus_loop: mimi.decode")
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        self._mark_progress("opus_loop: other_mimi.decode")
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        self._mark_progress("opus_loop: main_pcm.cpu()")
                        main_pcm = main_pcm.cpu()
                        self._mark_progress("opus_loop: opus_writer.append_pcm")
                        pcm_out = main_pcm[0, 0].numpy()
                        opus_writer.append_pcm(pcm_out)
                        out_sq_sum += float(np.square(pcm_out).sum())
                        out_sample_cnt += pcm_out.shape[-1]
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text_tokens_since_report += 1
                            last_text_time = time.time()
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            self._mark_progress("opus_loop: ws.send_bytes (text)")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

                    processing_time_since_report += time.time() - be
                    frames_since_report += 1
                    self._mark_progress("opus_loop: between frames")

                now = time.time()
                if now - last_report >= 5.0:
                    backlog_s = (0 if all_pcm_data is None else all_pcm_data.shape[-1]) / self.mimi.sample_rate
                    budget_s = frames_since_report * frame_wall_time
                    rtf = (processing_time_since_report / budget_s) if budget_s > 0 else 0.0
                    out_rms = (out_sq_sum / out_sample_cnt) ** 0.5 if out_sample_cnt > 0 else 0.0
                    clog.log(
                        "info" if rtf < 0.9 else "warning",
                        f"perf: {frames_since_report} frames in last {now - last_report:.1f}s, "
                        f"processing/real-time ratio={rtf:.2f}, unprocessed input backlog={backlog_s:.2f}s, "
                        f"text_tokens={text_tokens_since_report}, out_rms={out_rms:.4f}, "
                        f"last_text={now - last_text_time:.0f}s ago",
                    )
                    last_report = now
                    frames_since_report = 0
                    processing_time_since_report = 0.0
                    text_tokens_since_report = 0
                    out_sq_sum = 0.0
                    out_sample_cnt = 0

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                self._mark_progress("send_loop: opus_writer.read_bytes")
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    self._mark_progress("send_loop: ws.send_bytes (audio)")
                    await ws.send_bytes(b"\x01" + msg)

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
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Surface any exception that ended the session early instead of letting it
                # vanish silently (an un-retrieved task exception is otherwise only ever
                # logged much later, if at all, when the task object is garbage collected).
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        clog.log("error", f"session task {task.get_name()} failed: {exc!r}")
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
                             "0 disables recovery (default).")
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
    )
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
