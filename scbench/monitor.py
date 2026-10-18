"""Process-tree sampling of memory and CPU time while a workload runs."""

from __future__ import annotations

import bisect
import sys
import threading
import time

import psutil


def _memory(p: psutil.Process) -> int:
    """Proportional set size where available (Linux), so pages shared by forked
    workers are not counted once per worker; resident set size otherwise."""
    if sys.platform == "linux":
        try:
            with open(f"/proc/{p.pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return p.memory_info().rss


class TreeMonitor(threading.Thread):
    """Samples summed memory (PSS on Linux, else RSS) and cumulative CPU seconds of a process and all descendants.

    CPU time of a process is remembered after it exits, so short-lived workers
    (forked BiocParallel workers, numba/OpenMP threads are in-process anyway)
    still count as long as they were seen once.
    """

    def __init__(self, pid: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.root = psutil.Process(pid)
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (time, rss_bytes, cpu_seconds)
        self._cpu: dict[tuple[int, float], float] = {}
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.sample()
            self._stop_evt.wait(self.interval)

    def sample(self) -> None:
        try:
            procs = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            return
        rss = 0
        now = time.time()
        for p in procs:
            try:
                with p.oneshot():
                    rss += _memory(p)
                    ct = p.cpu_times()
                    self._cpu[(p.pid, p.create_time())] = ct.user + ct.system
            except psutil.Error:
                pass
        self.samples.append((now, float(rss), sum(self._cpu.values())))

    def stop(self) -> None:
        self._stop_evt.set()
        self.join()

    # -- analysis ---------------------------------------------------------
    def _cpu_at(self, t: float) -> float:
        times = [s[0] for s in self.samples]
        i = bisect.bisect_left(times, t)
        if i == 0:
            return self.samples[0][2]
        if i >= len(times):
            return self.samples[-1][2]
        (t0, _, c0), (t1, _, c1) = self.samples[i - 1], self.samples[i]
        return c0 + (c1 - c0) * (t - t0) / (t1 - t0) if t1 > t0 else c1

    def window(self, start: float, end: float) -> dict:
        """Peak tree memory (GB) and mean busy cores between two timestamps."""
        if not self.samples:
            return {"peak_rss_gb": None, "cpu_cores": None}
        inside = [s[1] for s in self.samples if start <= s[0] <= end]
        # Include the closest sample on either side so very short steps still get a value.
        times = [s[0] for s in self.samples]
        lo = max(bisect.bisect_left(times, start) - 1, 0)
        hi = min(bisect.bisect_right(times, end), len(times) - 1)
        peak = max(inside + [self.samples[lo][1], self.samples[hi][1]])
        dt = end - start
        cores = (self._cpu_at(end) - self._cpu_at(start)) / dt if dt > 2 * self.interval else None
        return {"peak_rss_gb": round(peak / 1e9, 3),
                "cpu_cores": round(cores, 2) if cores is not None else None}
