#!/usr/bin/env python3
"""
Scan local system resources and suggest sensible Ollama model defaults.

This is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Resources:
    cpu_cores: int
    ram_gb: float
    gpu_vram_gb: float | None
    gpu_name: str | None


def _read_first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def get_ram_gb() -> float:
    # Linux: /proc/meminfo
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = f.read()
        kb = _read_first(r"^MemTotal:\s+(\d+)\s+kB\s*$", data)
        if kb:
            return int(kb) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def get_cpu_cores() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def get_nvidia_gpu() -> tuple[float | None, str | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        # If multiple GPUs, just take the first line.
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts else None
        mem_mb = float(parts[1]) if len(parts) > 1 else None
        if mem_mb is None:
            return None, name
        return mem_mb / 1024.0, name
    except Exception:
        return None, None


def detect_resources() -> Resources:
    gpu_vram_gb, gpu_name = get_nvidia_gpu()
    return Resources(
        cpu_cores=get_cpu_cores(),
        ram_gb=get_ram_gb(),
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name,
    )


def recommend_chat_model(r: Resources) -> str:
    # Heuristics are conservative; Ollama can run larger models slowly, but this aims for "usable".
    vram = r.gpu_vram_gb
    if vram is not None:
        if vram >= 24:
            return "llama3.1:70b"
        if vram >= 12:
            return "llama3.1:8b"
        # nvidia-smi reports in MB; 8GB cards often show ~7.8GB
        if vram >= 7.5:
            # If you have enough RAM to spill, let 8B be an option; otherwise favor 3B for speed.
            if r.ram_gb >= 48:
                return "llama3.1:8b"
            return "llama3.2:3b"
        return "llama3.2:1b"

    # CPU/RAM-only
    if r.ram_gb >= 64 and r.cpu_cores >= 12:
        return "llama3.1:8b"
    if r.ram_gb >= 32 and r.cpu_cores >= 8:
        return "llama3.2:3b"
    return "llama3.2:1b"


def recommend_embed_model(r: Resources) -> str:
    vram = r.gpu_vram_gb
    if (vram is not None and vram >= 12) or r.ram_gb >= 16:
        return "mxbai-embed-large"
    return "nomic-embed-text"


def main() -> None:
    r = detect_resources()
    chat = recommend_chat_model(r)
    embed = recommend_embed_model(r)

    print("System:")
    print(f"  OS:   {platform.platform()}")
    print(f"  CPU:  {r.cpu_cores} cores")
    print(f"  RAM:  {r.ram_gb:.1f} GB")
    if r.gpu_vram_gb is not None:
        print(f"  GPU:  {r.gpu_name or 'NVIDIA'} ({r.gpu_vram_gb:.1f} GB VRAM)")
    else:
        print("  GPU:  (none detected via nvidia-smi)")

    print("\nSuggested models:")
    print(f"  CHAT_MODEL={chat}")
    print(f"  EMBED_MODEL={embed}")

    print("\nPull via docker compose:")
    print(f"  docker compose exec -it ollama ollama pull {chat}")
    print(f"  docker compose exec -it ollama ollama pull {embed}")

    print("\nAdd to .env:")
    print(f"  CHAT_MODEL={chat}")
    print(f"  EMBED_MODEL={embed}")


if __name__ == "__main__":
    main()

