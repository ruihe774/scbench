"""scbench scanpy workload: a standard scanpy analysis of one 10x sample or an atlas.

Each timed step appends {"step", "start", "end"} (epoch seconds) to the events
file so the runner can attribute wall time, CPU use and memory to steps.
"""

import time

T0 = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="dataset directory containing manifest.json")
p.add_argument("--mode", choices=["single", "atlas"], required=True)
p.add_argument("--threads", type=int, required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--events", required=True)
args = p.parse_args()

data_dir, outdir = Path(args.data), Path(args.outdir)
events = open(args.events, "a", buffering=1)


@contextmanager
def step(name, start=None):
    t = time.time() if start is None else start
    yield
    events.write(json.dumps({"step": name, "start": t, "end": time.time()}) + "\n")


with step("load_libraries", start=T0):
    import anndata as ad
    import harmonypy
    import numpy as np
    import scanpy as sc

    sc.settings.n_jobs = args.threads
    sc.settings.verbosity = 0

manifest = json.loads((data_dir / "manifest.json").read_text())
samples = manifest["samples"]
atlas = args.mode == "atlas"
batch_key = "sample" if atlas else None

with step("read_data"):
    parts = {}
    for s in samples:
        a = sc.read_10x_h5(data_dir / s["path"])
        a.var_names_make_unique()
        a.obs["barcode"] = a.obs_names
        parts[s["name"]] = a
    if atlas:
        adata = ad.concat(parts, label="sample", index_unique="_", merge="same")
    else:
        (name, adata), = parts.items()
        adata.obs["sample"] = name
    del parts
n_input = adata.n_obs

with step("qc_metrics"):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

with step("qc_filter"):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

with step("doublet_detection"):
    sc.pp.scrublet(adata, batch_key=batch_key, random_state=0)
    n_doublets = int(adata.obs["predicted_doublet"].sum())
    adata = adata[~adata.obs["predicted_doublet"].to_numpy()].copy()

with step("normalize"):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

with step("hvg"):
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=batch_key)

with step("scale"):
    hvg = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(hvg, max_value=10)

with step("pca"):
    sc.pp.pca(hvg, n_comps=50, random_state=0)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    del hvg

rep = "X_pca"
if atlas:
    with step("integration"):
        # Called directly: scanpy's harmony_integrate wrapper predates harmonypy 2's (cells x PCs) output.
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, "sample", random_state=0,
                                   ncores=args.threads, verbose=False)
        z = np.asarray(ho.Z_corr)
        adata.obsm["X_pca_harmony"] = z if z.shape[0] == adata.n_obs else z.T
        rep = "X_pca_harmony"

with step("neighbors"):
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep=rep, random_state=0)

with step("clustering"):
    sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, key_added="cluster",
                 random_state=0)

with step("umap"):
    sc.tl.umap(adata, random_state=0)

with step("markers"):
    sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon")

with step("save"):
    adata.write_h5ad(outdir / "result.h5ad")

events.close()
adata.obs[["sample", "barcode", "cluster"]].to_csv(outdir / "clusters.csv.gz", index=False)


def version(mod):
    try:
        from importlib.metadata import version as v
        return v(mod)
    except Exception:
        return None


try:
    from threadpoolctl import threadpool_info
    blas = [f"{c['internal_api']} {c.get('version')} {c.get('threading_layer', '')} {c.get('architecture', '')}".strip()
            for c in threadpool_info() if c.get("user_api") == "blas"]
except Exception:
    blas = None

info = {
    "framework": "scanpy",
    "versions": {"python": sys.version.split()[0],
                 **{m: version(m) for m in ["scanpy", "anndata", "numpy", "scipy", "scikit-learn",
                                            "numba", "umap-learn", "pynndescent", "leidenalg",
                                            "igraph", "harmonypy", "pandas", "h5py"]}},
    "blas": blas,
    "metrics": {
        "n_cells_input": int(n_input),
        "n_doublets_removed": n_doublets,
        "n_cells_final": int(adata.n_obs),
        "n_genes_final": int(adata.n_vars),
        "n_hvg": int(adata.var["highly_variable"].sum()),
        "n_clusters": int(adata.obs["cluster"].nunique()),
    },
}
(outdir / "info.json").write_text(json.dumps(info, indent=2))
