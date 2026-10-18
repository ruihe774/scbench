"""scbench command line: generate data, run workloads, report and compare results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from scbench import __version__
from scbench.generate import SIZES, generate
from scbench.monitor import TreeMonitor
from scbench.sysinfo import system_info

ROOT = Path(__file__).resolve().parent.parent

# variant -> (pixi environment, dataset kind, command)
VARIANTS = {
    "scanpy-single": ("scanpy", "single", ["python", "-W", "ignore", "workloads/scanpy_workload.py"]),
    "seurat-single": ("seurat", "single", ["Rscript", "workloads/seurat_workload.R"]),
    "scanpy-atlas": ("scanpy", "atlas", ["python", "-W", "ignore", "workloads/scanpy_workload.py"]),
    "seurat-atlas": ("seurat", "atlas", ["Rscript", "workloads/seurat_workload.R"]),
}
STEPS = ["load_libraries", "read_data", "qc_metrics", "qc_filter", "doublet_detection", "normalize",
         "hvg", "scale", "pca", "integration", "neighbors", "clustering", "umap", "markers", "save"]
THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
               "RCPP_PARALLEL_NUM_THREADS"]
# Rough peak memory need (GB) of the heaviest variant per size, used for a warning only.
MEMORY_HINT_GB = {"tiny": 4, "standard": 32, "large": 96}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# -- environments ------------------------------------------------------------
_env_cache: dict[str, dict] = {}


def pixi_env(name: str) -> dict:
    """Activated environment variables of a pixi environment (installs it if needed)."""
    if name not in _env_cache:
        out = subprocess.run(["pixi", "shell-hook", "--json", "-e", name, "--manifest-path",
                              str(ROOT / "pixi.toml")], check=True, capture_output=True, text=True)
        env = dict(os.environ)
        env.update(json.loads(out.stdout)["environment_variables"])
        if name == "seurat":
            _ensure_bioc_data(Path(env["CONDA_PREFIX"]), env)
        _env_cache[name] = env
    return _env_cache[name]


def _ensure_bioc_data(prefix: Path, env: dict) -> None:
    """Bioconda data packages (GenomeInfoDbData, needed by scDblFinder) download their
    payload in a post-link script, which pixi does not run. Run it once ourselves."""
    for script in sorted((prefix / "bin").glob(".bioconductor-*data-post-link.sh")):
        pkg = script.name.removeprefix(".bioconductor-").removesuffix("-post-link.sh")
        if any(p.name.lower() == pkg for p in (prefix / "lib/R/library").iterdir()):
            continue
        log(f"installing Bioconductor data package {pkg} (one-time) ...")
        subprocess.run(["bash", str(script)], env={**env, "PREFIX": str(prefix)}, check=True,
                       stdout=subprocess.DEVNULL)


# -- validation --------------------------------------------------------------
def adjusted_rand_index(a, b) -> float:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sum_ij = comb(table).sum()
    sum_a, sum_b = comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = sum_a * sum_b / comb(table.sum())
    max_index = (sum_a + sum_b) / 2
    return float((sum_ij - expected) / (max_index - expected)) if max_index != expected else 1.0


def validate(data_dir: Path, clusters_csv: Path) -> dict:
    """Compare a workload's clustering with the simulated ground truth."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    truth = {}
    for s in manifest["samples"]:
        with gzip.open(data_dir / s["truth"], "rt") as fh:
            for row in csv.DictReader(fh):
                truth[(s["name"], row["barcode"])] = row
    with gzip.open(clusters_csv, "rt") as fh:
        rows = list(csv.DictReader(fh))
    kept = [(truth[(r["sample"], r["barcode"])], r["cluster"]) for r in rows]
    singlets = [(t["cell_type"], t["lineage"], c) for t, c in kept if t["is_doublet"] == "0"]
    cell_type, lineage, cluster = zip(*singlets)
    n_doublets = sum(t["is_doublet"] == "1" for t in truth.values())
    n_doublets_kept = sum(t["is_doublet"] == "1" for t, _ in kept)
    return {
        "ari_cell_type": round(adjusted_rand_index(cell_type, cluster), 3),
        "ari_lineage": round(adjusted_rand_index(lineage, cluster), 3),
        "doublets_removed_frac": round(1 - n_doublets_kept / n_doublets, 3) if n_doublets else None,
    }


# -- running -----------------------------------------------------------------
def run_variant(variant: str, data_dir: Path, run_dir: Path, threads: int, keep_outputs: bool) -> dict:
    env_name, kind, cmd = VARIANTS[variant]
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "events.jsonl"
    env = dict(pixi_env(env_name))
    env.update({v: str(threads) for v in THREAD_VARS})
    env["PYTHONHASHSEED"] = "0"
    argv = cmd + ["--data", str(data_dir), "--mode", kind, "--threads", str(threads),
                  "--outdir", str(run_dir), "--events", str(events)]

    with open(run_dir / "workload.log", "w") as logf:
        t_start = time.time()
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        mon = TreeMonitor(proc.pid)
        mon.start()
        _, status, rusage = os.wait4(proc.pid, 0)
        t_end = time.time()
        mon.stop()
    proc.returncode = os.waitstatus_to_exitcode(status)

    steps = {}
    if events.exists():
        for line in events.read_text().splitlines():
            e = json.loads(line)
            steps[e["step"]] = {"seconds": round(e["end"] - e["start"], 3), **mon.window(e["start"], e["end"])}
    with open(run_dir / "timeline.csv", "w") as fh:
        fh.write("time,rss_gb,cpu_seconds\n")
        for t, rss, cpu in mon.samples:
            fh.write(f"{t - t_start:.2f},{rss / 1e9:.3f},{cpu:.2f}\n")

    result = {
        "status": "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})",
        "total_seconds": round(sum(s["seconds"] for s in steps.values()), 3),
        "process_wall_seconds": round(t_end - t_start, 3),
        "cpu_seconds": round(rusage.ru_utime + rusage.ru_stime, 1),
        "peak_rss_gb": round(max((s[1] for s in mon.samples), default=0) / 1e9, 3),
        "steps": steps,
    }
    info_path = run_dir / "info.json"
    if proc.returncode == 0 and info_path.exists():
        result["info"] = json.loads(info_path.read_text())
        try:
            result["validation"] = validate(data_dir, run_dir / "clusters.csv.gz")
        except Exception as exc:  # validation must never hide timing results
            result["validation"] = {"error": repr(exc)}
    if not keep_outputs:
        for f in ("result.h5ad", "result.rds"):
            (run_dir / f).unlink(missing_ok=True)
    return result


def summarize(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "ok"]
    if not ok:
        return {}
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
    steps = {}
    for name in STEPS:
        vals = [r["steps"][name] for r in ok if name in r["steps"]]
        if vals:
            steps[name] = {k: med([v[k] for v in vals if v[k] is not None])
                           for k in ("seconds", "peak_rss_gb", "cpu_cores")}
    return {
        "total_seconds": med([r["total_seconds"] for r in ok]),
        "total_seconds_min": min(r["total_seconds"] for r in ok),
        "total_seconds_max": max(r["total_seconds"] for r in ok),
        "peak_rss_gb": max(r["peak_rss_gb"] for r in ok),
        "cpu_seconds": med([r["cpu_seconds"] for r in ok]),
        "steps": steps,
    }


def cmd_run(a) -> None:
    variants = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variant(s): {', '.join(sorted(unknown))}; choose from {', '.join(VARIANTS)}")
    threads = a.threads or psutil.cpu_count(logical=False) or os.cpu_count()
    sysinfo = system_info(ROOT)
    if sysinfo["memory_gb"] < MEMORY_HINT_GB[a.size]:
        log(f"warning: {sysinfo['memory_gb']} GB RAM; size '{a.size}' may need ~{MEMORY_HINT_GB[a.size]} GB")
    if sysinfo["load_avg_at_start"] and sysinfo["load_avg_at_start"][0] > 0.25 * sysinfo["cpu_logical_cores"]:
        log(f"warning: system load is {sysinfo['load_avg_at_start'][0]}; results may be noisy")

    datasets = {}
    for kind in sorted({VARIANTS[v][1] for v in variants}):
        datasets[kind] = generate(Path(a.data_dir), a.size, kind, a.seed, log=log)
    for env_name in sorted({VARIANTS[v][0] for v in variants}):
        log(f"preparing pixi environment '{env_name}' ...")
        pixi_env(env_name)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.results_dir) / f"{stamp}-{sysinfo['hostname']}-{a.size}"
    out.mkdir(parents=True)
    results = {
        "scbench_version": __version__,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": sysinfo,
        "config": {"size": a.size, "seed": a.seed, "threads": threads, "repeats": a.repeats,
                   "datasets": {k: {"n_cells": json.loads((d / "manifest.json").read_text())["n_cells_total"],
                                    "n_samples": len(json.loads((d / "manifest.json").read_text())["samples"]),
                                    "fingerprint": json.loads((d / "manifest.json").read_text())["fingerprint"]}
                                for k, d in datasets.items()}},
        "variants": {},
    }
    log(f"\n{sysinfo['cpu_model']} | {sysinfo['cpu_physical_cores']} cores / "
        f"{sysinfo['cpu_logical_cores']} threads | {sysinfo['memory_gb']} GB | using {threads} threads")
    log(f"results -> {out}\n")
    for v in variants:
        runs = []
        for rep in range(1, a.repeats + 1):
            log(f"[{v}] repeat {rep}/{a.repeats} ...")
            r = run_variant(v, datasets[VARIANTS[v][1]], out / v / f"rep{rep}", threads, a.keep_outputs)
            runs.append(r)
            extra = ""
            if "validation" in r and "ari_cell_type" in r["validation"]:
                extra = f", ARI {r['validation']['ari_cell_type']}"
            log(f"[{v}]   {r['status']}: {r['total_seconds']:.1f} s, peak {r['peak_rss_gb']:.1f} GB{extra}")
            if r["status"] != "ok":
                log(f"[{v}]   see {out / v / f'rep{rep}' / 'workload.log'}")
        results["variants"][v] = {"runs": runs, "summary": summarize(runs)}
        (out / "results.json").write_text(json.dumps(results, indent=2))
    log("")
    print_report(results)
    log(f"\nsaved {out / 'results.json'}")


# -- reporting ---------------------------------------------------------------
def _fmt(x, spec):
    if x is not None:
        return format(x, spec)
    width = "".join(ch for ch in spec.split(".")[0] if ch.isdigit())
    return "-".rjust(int(width or 0))


def print_report(results: dict) -> None:
    s, c = results["system"], results["config"]
    log(f"scbench {results['scbench_version']} | {results['created']} | size={c['size']} threads={c['threads']}"
        f" repeats={c['repeats']}")
    log(f"{s['cpu_model']} | {s['cpu_physical_cores']}C/{s['cpu_logical_cores']}T | {s['memory_gb']} GB | {s['os']}")
    for kind, d in c["datasets"].items():
        log(f"  {kind} dataset: {d['n_cells']:,} cells in {d['n_samples']} sample(s)")
    for v, vr in results["variants"].items():
        sm = vr["summary"]
        log("")
        if not sm:
            log(f"== {v}: FAILED ({vr['runs'][0]['status']})")
            continue
        val = next((r.get("validation", {}) for r in vr["runs"] if r["status"] == "ok"), {})
        info = next((r.get("info", {}) for r in vr["runs"] if r["status"] == "ok"), {})
        m = info.get("metrics", {})
        log(f"== {v}: {sm['total_seconds']:.1f} s (median)  peak {sm['peak_rss_gb']:.1f} GB  "
            f"CPU {sm['cpu_seconds']:.0f} s  | {m.get('n_cells_final', '?'):,} cells, "
            f"{m.get('n_clusters', '?')} clusters, ARI {val.get('ari_cell_type', '-')}, "
            f"doublets removed {_fmt(val.get('doublets_removed_frac'), '.0%')}")
        log(f"   {'step':<18}{'seconds':>10}{'share':>8}{'cores':>8}{'peak GB':>9}")
        for name, st in sm["steps"].items():
            share = st["seconds"] / sm["total_seconds"] if sm["total_seconds"] else 0
            log(f"   {name:<18}{st['seconds']:>10.2f}{share:>8.1%}{_fmt(st['cpu_cores'], '>8.1f')}"
                f"{_fmt(st['peak_rss_gb'], '>9.2f')}")


def _load(path: str) -> dict:
    p = Path(path)
    return json.loads((p / "results.json" if p.is_dir() else p).read_text())


def cmd_report(a) -> None:
    print_report(_load(a.results))


def cmd_compare(a) -> None:
    res = [_load(p) for p in a.results]
    labels = [f"#{i + 1}" for i in range(len(res))]
    for lab, r in zip(labels, res):
        s, c = r["system"], r["config"]
        log(f"{lab}: {s['hostname']} | {s['cpu_model']} | {s['cpu_physical_cores']}C/{s['cpu_logical_cores']}T"
            f" | {s['memory_gb']} GB | size={c['size']} threads={c['threads']} | {r['created']}")
    sizes = {r["config"]["size"] for r in res}
    if len(sizes) > 1:
        log(f"warning: comparing different dataset sizes {sorted(sizes)}")
    variants = [v for v in VARIANTS if any(v in r["variants"] for r in res)]
    log("")
    w = 14
    log(f"{'total seconds':<22}" + "".join(f"{lab:>{w}}" for lab in labels) + "   speedup vs #1")
    totals = []
    for v in variants:
        row = [r["variants"].get(v, {}).get("summary", {}).get("total_seconds") for r in res]
        totals.append(row)
        sp = " ".join(_fmt(row[0] / x, ".2fx") if row[0] and x else "-" for x in row[1:])
        log(f"{v:<22}" + "".join(_fmt(x, f">{w}.1f") for x in row) + f"   {sp}")
    geo = []
    for i in range(len(res)):
        col = [t[i] for t in totals]
        geo.append(math.exp(statistics.fmean(math.log(x) for x in col)) if all(col) else None)
    log(f"{'geometric mean':<22}" + "".join(_fmt(x, f">{w}.1f") for x in geo) + "   " +
        " ".join(_fmt(geo[0] / x, ".2fx") if geo[0] and x else "-" for x in geo[1:]))
    if a.steps:
        for v in variants:
            log(f"\n{v} step seconds")
            for name in STEPS:
                row = [r["variants"].get(v, {}).get("summary", {}).get("steps", {}).get(name, {}).get("seconds")
                       for r in res]
                if any(x is not None for x in row):
                    log(f"  {name:<20}" + "".join(_fmt(x, f">{w}.2f") for x in row))


def cmd_generate(a) -> None:
    for kind in (["single", "atlas"] if a.kind == "both" else [a.kind]):
        d = generate(Path(a.data_dir), a.size, kind, a.seed, log=log)
        log(f"{kind}: {d}")


def cmd_clean(a) -> None:
    for p in (Path(a.data_dir),):
        if p.exists():
            shutil.rmtree(p)
            log(f"removed {p}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="scbench", description=__doc__)
    p.add_argument("--version", action="version", version=f"scbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--size", choices=list(SIZES), default="standard",
                        help="dataset size: " + "; ".join(f"{k}: single {s:,} cells, atlas {n}x~{m:,} cells"
                                                         for k, (s, n, m) in SIZES.items()))
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--data-dir", default=str(ROOT / "data"))

    r = sub.add_parser("run", help="run benchmark variants")
    common(r)
    r.add_argument("--variants", default="all", help=f"comma-separated subset of: {', '.join(VARIANTS)}")
    r.add_argument("--threads", type=int, default=None, help="threads per workload (default: physical cores)")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--results-dir", default=str(ROOT / "results"))
    r.add_argument("--keep-outputs", action="store_true", help="keep result.h5ad / result.rds files")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("generate", help="generate synthetic datasets only")
    common(g)
    g.add_argument("--kind", choices=["single", "atlas", "both"], default="both")
    g.set_defaults(fn=cmd_generate)

    rp = sub.add_parser("report", help="print a results.json (or results directory)")
    rp.add_argument("results")
    rp.set_defaults(fn=cmd_report)

    c = sub.add_parser("compare", help="compare several results (e.g. from different machines)")
    c.add_argument("results", nargs="+")
    c.add_argument("--steps", action="store_true", help="also show per-step times")
    c.set_defaults(fn=cmd_compare)

    cl = sub.add_parser("clean-data", help="delete generated datasets")
    cl.add_argument("--data-dir", default=str(ROOT / "data"))
    cl.set_defaults(fn=cmd_clean)

    a = p.parse_args(argv)
    a.fn(a)
