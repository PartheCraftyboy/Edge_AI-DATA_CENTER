"""
One vision-language model in VRAM at a time.

Measured on the RTX 4050, which has 6141 MiB:

    baseline                        40 MiB used
    qwen2.5vl:3b via Ollama       4196 MiB used,  1576 free
    PaddleOCR-VL on top of it       OOM, 9 MiB free

So they do not both fit, and the gap is not close enough to tune around.
Note that Ollama's own `/api/ps` reports 2764 MiB for that model — it counts
the weights and not the CUDA context or the KV cache, so trusting it would
have suggested a comfortable 1.4 GB of headroom that does not exist.

This matters because the two models now have different jobs and are reached
from different places. PaddleOCR-VL reads, on the compile path, in the
server. Qwen judges region classes, offline, in `app.distil`. Nothing stops
those running at the same time — leaving the server up and starting a
training pass is the obvious thing to do — and the failure when they do is a
CUDA OOM in whichever asked second, which surfaces as a held region or a
dead training run rather than as anything that names the cause.

So residency is explicit. A model about to load calls `acquire`, which
evicts whoever holds VRAM first. The cost is a reload, about four seconds
for Ollama and eight for Paddle, paid only when the two actually alternate.
That is the right trade: a slow compile is a nuisance, an OOM mid-run is an
evening.

This is a within-process registry and deliberately not a lock file. It
cannot arbitrate between the server and a separate `app.distil` process, and
pretending otherwise with a lock would add failure modes (a stale lock after
a crash) worse than the one being solved. What it does instead is make each
process evict whatever it finds resident, including another process's
Ollama model, which is enough: Ollama is a shared server and unloading it is
a request any client may make.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_resident: dict[str, object] = {}


def acquire(name: str, release) -> None:
    """Claim VRAM for `name`, evicting any other resident model first.

    `release` is how this model is later unloaded. Re-acquiring a model that
    already holds VRAM does nothing, so this is cheap to call before every
    inference rather than only on first load.
    """
    with _lock:
        if name in _resident:
            return
        others = list(_resident.items())
        _resident.clear()
        for _, free_it in others:
            try:
                free_it()
            except Exception:
                # An eviction that fails must not stop the model that asked.
                # The worst case is the OOM we were avoiding, which the
                # caller already handles.
                pass
        _resident[name] = release


def release(name: str) -> None:
    """Give up VRAM held under `name`, without unloading anything else."""
    with _lock:
        free_it = _resident.pop(name, None)
    if free_it is not None:
        try:
            free_it()
        except Exception:
            pass


def resident() -> list[str]:
    """Which models this process believes are loaded. For tests and status."""
    with _lock:
        return sorted(_resident)


def forget(name: str) -> None:
    """Drop the record without calling its release, for a model that has
    already unloaded itself."""
    with _lock:
        _resident.pop(name, None)
