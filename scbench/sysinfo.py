"""Describe the machine a benchmark ran on."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import psutil


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def system_info(root: Path) -> dict:
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.max or f.current) if f else None
    except Exception:
        pass
    lock = root / "pixi.lock"
    try:
        pixi_version = subprocess.check_output(["pixi", "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pixi_version = None
    return {
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_mhz": freq,
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "machine": platform.machine(),
        "load_avg_at_start": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
        "pixi": pixi_version,
        "pixi_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()[:16] if lock.exists() else None,
    }
