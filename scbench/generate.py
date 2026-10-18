"""Synthetic 10x Chromium (3' v3) datasets.

Writes CellRanger-style ``filtered_feature_bc_matrix.h5`` files whose layout,
dtypes and statistical properties resemble real filtered 10x samples:

* 36,601 features (size of the GRCh38-2020-A reference) with Ensembl-like ids,
  real names for mitochondrial / ribosomal / hemoglobin / sex / housekeeping
  genes, lncRNA-style names, and a few duplicated symbols.
* PBMC-like cell type hierarchy (lineage -> subtype) with marker genes.
* Log-normal library sizes, negative-binomial (gamma-Poisson) counts with
  gene-specific overdispersion, mitochondrial fraction, ambient RNA,
  low-quality cells and heterotypic doublets at the 10x rate (~0.8% / 1k cells).
* Atlas samples additionally get donor sex, cell type composition shifts,
  sequencing depth differences and gene-wise batch effects.

Ground truth labels are written alongside each sample (``truth.csv.gz``) so the
runner can check that every workload produced a sensible clustering.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

N_FEATURES = 36601
GENOME = "GRCh38"
GENERATOR_VERSION = 2

SIZES = {
    # name: (cells in single sample, atlas sample count, mean cells per atlas sample)
    "tiny": (2_000, 4, 2_000),
    "standard": (10_000, 12, 8_000),
    "large": (20_000, 32, 10_000),
}

MT_GENES = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3",
            "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB"]
MT_WEIGHTS = [6, 4, 14, 10, 2, 9, 12, 5, 2, 8, 2, 1, 7]
RIBO_GENES = (
    [f"RPL{i}" for i in ["3", "4", "5", "6", "7", "7A", "8", "9", "10", "10A", "11", "12", "13",
                         "13A", "14", "15", "17", "18", "18A", "19", "21", "22", "23", "23A", "24",
                         "26", "27", "27A", "28", "29", "30", "31", "32", "34", "35", "35A", "36",
                         "36A", "37", "37A", "38", "39", "41", "P0", "P1", "P2"]]
    + [f"RPS{i}" for i in ["A", "2", "3", "3A", "4X", "5", "6", "7", "8", "9", "10", "11", "12",
                           "13", "14", "15", "15A", "16", "17", "18", "19", "20", "21", "23", "24",
                           "25", "26", "27", "27A", "28", "29"]]
)
HB_GENES = ["HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ", "HBE1"]
HOUSEKEEPING = {"MALAT1": 25, "TMSB4X": 12, "B2M": 12, "EEF1A1": 10, "FTL": 6, "ACTB": 8,
                "FTH1": 6, "TPT1": 6, "MYL6": 4, "TMSB10": 4, "PFN1": 3, "CFL1": 3, "GAPDH": 3,
                "ACTG1": 3, "NEAT1": 3, "UBC": 2, "OAZ1": 2, "SRGN": 2, "HLA-A": 2, "HLA-B": 3,
                "HLA-C": 2}
FEMALE_GENES = ["XIST", "TSIX"]
MALE_GENES = ["RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "USP9Y", "ZFY"]

# Continuous gene programmes shared across cell types (within-cluster heterogeneity):
# name, genes, fraction of cells with the programme active, max share of the library
PROGRAMS = [
    ("cell_cycle", 150, 0.06, 0.05),
    ("interferon", 60, 0.30, 0.03),
    ("stress", 40, 0.40, 0.02),
    ("activation", 100, 0.50, 0.03),
]

# name, lineage, abundance, library size factor, ribosomal factor
CELL_TYPES = [
    ("CD4 T naive", "T", 0.17, 0.80, 1.5),
    ("CD4 T memory", "T", 0.13, 0.85, 1.3),
    ("Treg", "T", 0.02, 0.85, 1.2),
    ("CD8 T naive", "T", 0.07, 0.80, 1.5),
    ("CD8 T effector", "T", 0.07, 0.90, 1.1),
    ("MAIT", "T", 0.02, 0.90, 1.1),
    ("NK", "NK", 0.07, 0.90, 0.9),
    ("B naive", "B", 0.07, 0.90, 1.2),
    ("B memory", "B", 0.04, 0.95, 1.1),
    ("Plasmablast", "B", 0.01, 2.50, 0.5),
    ("CD14 Mono", "Myeloid", 0.18, 1.30, 0.6),
    ("CD16 Mono", "Myeloid", 0.05, 1.20, 0.6),
    ("cDC2", "Myeloid", 0.02, 1.40, 0.7),
    ("pDC", "pDC", 0.01, 1.00, 0.8),
    ("Platelet", "Platelet", 0.01, 0.30, 0.3),
    ("Erythrocyte", "Erythroid", 0.005, 0.40, 0.2),
]


@dataclass
class Universe:
    """Everything shared by all samples generated with one seed."""

    gene_ids: np.ndarray
    gene_names: np.ndarray
    profiles: np.ndarray  # (n_types, n_genes) non-mito expression, rows sum to 1
    programs: np.ndarray  # (n_programs, n_genes) additive programme profiles, rows sum to 1
    mito: np.ndarray  # (n_genes,) mitochondrial profile, sums to 1
    dispersion_shape: np.ndarray  # (n_genes,) gamma shape = 1 / overdispersion
    female_idx: np.ndarray
    male_idx: np.ndarray
    type_props: np.ndarray
    type_lib: np.ndarray


def build_universe(seed: int) -> Universe:
    rng = np.random.default_rng([seed, 0])

    special = MT_GENES + RIBO_GENES + HB_GENES + list(HOUSEKEEPING) + FEMALE_GENES + MALE_GENES
    n_rest = N_FEATURES - len(special)
    n_lnc = int(n_rest * 0.42)
    n_pc = n_rest - n_lnc
    pc_names = [f"SG{i:05d}" for i in range(1, n_pc + 1)]
    lnc_nums = rng.choice(np.arange(100_000, 1_000_000), size=n_lnc, replace=False)
    lnc_names = [f"{rng.choice(['AC', 'AL', 'AP'])}{n}.{rng.integers(1, 4)}" for n in lnc_nums]
    # A handful of duplicated symbols, as in real references (exercises make-unique paths).
    for i in rng.choice(n_pc, size=10, replace=False):
        pc_names[i] = pc_names[(i + 1) % n_pc]

    nonmito = np.array(pc_names + lnc_names + RIBO_GENES + HB_GENES + list(HOUSEKEEPING)
                       + FEMALE_GENES + MALE_GENES, dtype=object)
    kind = np.concatenate([np.full(n_pc, 1), np.full(n_lnc, 2), np.zeros(len(nonmito) - n_rest, int)])
    order = rng.permutation(len(nonmito))  # genomic order is effectively random; chrM last
    names = np.concatenate([nonmito[order], np.array(MT_GENES, dtype=object)])
    kind = np.concatenate([kind[order], np.zeros(len(MT_GENES), int)])
    ids = np.array([f"ENSG{n:011d}" for n in np.sort(rng.choice(10**6, N_FEATURES, replace=False)) + 10**5],
                   dtype=object)
    idx = {n: i for i, n in enumerate(names)}  # last occurrence wins for duplicates; fine
    pos = lambda genes: np.array([idx[g] for g in genes])  # noqa: E731

    is_pc, is_lnc = kind == 1, kind == 2
    ribo, hb, mt = pos(RIBO_GENES), pos(HB_GENES), pos(MT_GENES)

    # Baseline relative abundance: heavy-tailed, many genes essentially silent.
    base = np.zeros(N_FEATURES)
    base[is_pc] = rng.lognormal(0.0, 1.7, is_pc.sum())
    base[is_lnc] = rng.lognormal(-2.8, 1.6, is_lnc.sum())
    silent = rng.random(N_FEATURES) < 0.15
    base[silent & (is_pc | is_lnc)] *= 1e-3
    base *= 0.66 / base.sum()
    ribo_w = rng.lognormal(0, 0.35, len(ribo))
    hk_w = np.array(list(HOUSEKEEPING.values()), float)

    n_types = len(CELL_TYPES)
    lineages = sorted({t[1] for t in CELL_TYPES})
    candidates = np.flatnonzero(is_pc & ~silent)
    log_prof = np.tile(np.log(base + 1e-12), (n_types, 1))

    lineage_effects = {}
    for lin in lineages:
        up = rng.choice(candidates, 300, replace=False)
        down = rng.choice(candidates, 150, replace=False)
        lineage_effects[lin] = (up, rng.normal(2.5, 0.8, 300), down, -np.abs(rng.normal(1.5, 0.5, 150)))
    for t, (_, lin, *_rest) in enumerate(CELL_TYPES):
        up, lfc_up, down, lfc_down = lineage_effects[lin]
        log_prof[t, up] += lfc_up
        log_prof[t, down] += lfc_down
        # Subtype programme: modest shifts plus a few specific markers.
        sub = rng.choice(candidates, 50, replace=False)
        log_prof[t, sub] += rng.normal(0.9, 0.5, 50)
        strong = rng.choice(np.flatnonzero(is_pc & silent), 3, replace=False)
        log_prof[t, strong] = np.log(base.max() * rng.uniform(0.01, 0.1, 3))
    prof = np.exp(log_prof)

    for t, (name, _, _, _, ribo_factor) in enumerate(CELL_TYPES):
        prof[t] *= 0.66 / prof[t].sum()
        prof[t, ribo] = 0.22 * ribo_factor * ribo_w / ribo_w.sum()
        prof[t, pos(HOUSEKEEPING)] = 0.12 * hk_w / hk_w.sum()
        prof[t, hb] = (0.5 if name == "Erythrocyte" else 2e-5) * rng.dirichlet(np.ones(len(hb)) * 2)
        prof[t, mt] = 0
        prof[t] /= prof[t].sum()

    programs = np.zeros((len(PROGRAMS), N_FEATURES))
    for k, (_, n, _, _) in enumerate(PROGRAMS):
        genes = rng.choice(candidates, n, replace=False)
        programs[k, genes] = rng.lognormal(0, 1, n)
        programs[k] /= programs[k].sum()

    mito = np.zeros(N_FEATURES)
    mito[mt] = np.array(MT_WEIGHTS, float) / sum(MT_WEIGHTS)

    phi = np.clip(rng.lognormal(np.log(0.12), 0.6, N_FEATURES), 0.01, 3.0)
    props = np.array([t[2] for t in CELL_TYPES])
    return Universe(
        gene_ids=ids, gene_names=names, profiles=prof, programs=programs, mito=mito, dispersion_shape=1.0 / phi,
        female_idx=pos(FEMALE_GENES), male_idx=pos(MALE_GENES),
        type_props=props / props.sum(), type_lib=np.array([t[3] for t in CELL_TYPES]),
    )


def _barcodes(rng: np.random.Generator, n: int) -> np.ndarray:
    seen: set[bytes] = set()
    out = []
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    while len(out) < n:
        for row in alphabet[rng.integers(0, 4, size=(n, 16))]:
            bc = row.tobytes() + b"-1"
            if bc not in seen:
                seen.add(bc)
                out.append(bc)
                if len(out) == n:
                    break
    return np.array(out, dtype="S18")


def simulate_sample(u: Universe, n_cells: int, seed: int, sample_idx: int, batch: bool,
                    median_umi: float = 7000.0, chunk: int = 1000):
    """Returns (csr counts cells x genes int32, barcodes, truth dict)."""
    rng = np.random.default_rng([seed, 1, sample_idx])
    n_types, n_genes = u.profiles.shape
    profiles = u.profiles.copy()

    props = u.type_props
    depth = 1.0
    if batch:
        # Donor/sample-level variation: composition, sex, depth, gene-wise batch effects.
        props = rng.dirichlet(u.type_props * 60 + 0.05)
        depth = rng.lognormal(0, 0.25)
        effect = rng.normal(0, 0.15, n_genes)
        strong = rng.random(n_genes) < 0.05
        effect[strong] += rng.normal(0, 0.8, strong.sum())
        profiles *= np.exp(effect)
        female = rng.random() < 0.5
    else:
        female = True
    sex_on, sex_off = (u.female_idx, u.male_idx) if female else (u.male_idx, u.female_idx)
    profiles[:, sex_on] = rng.uniform(2e-4, 1e-3, len(sex_on))
    profiles[:, sex_off] = 0
    profiles /= profiles.sum(1, keepdims=True)
    ambient = props @ profiles

    # Basis rows: cell types, ambient profile, mitochondrial profile, gene programmes.
    basis = np.vstack([profiles, ambient, u.mito, u.programs]).astype(np.float32)
    n_basis = basis.shape[0]

    n_doublets = int(round(n_cells * min(0.008 * n_cells / 1000, 0.25)))
    ctype = rng.choice(n_types, n_cells, p=props)
    low_q = rng.random(n_cells) < 0.05
    lib = rng.lognormal(np.log(median_umi * depth * u.type_lib[ctype]), 0.5)
    lib[low_q] *= 0.35
    mito_frac = rng.beta(4, 70, n_cells)
    mito_frac[low_q] = rng.beta(5, 8, low_q.sum())
    ambient_frac = rng.beta(2, 60, n_cells)

    coef = np.zeros((n_cells, n_basis), np.float32)
    rows = np.arange(n_cells)
    coef[rows, ctype] += lib * (1 - mito_frac) * (1 - ambient_frac)
    coef[:, n_types] += lib * (1 - mito_frac) * ambient_frac
    coef[:, n_types + 1] += lib * mito_frac
    for k, (_, _, active, strength) in enumerate(PROGRAMS):
        if batch:  # programme prevalence differs between donors (e.g. interferon response)
            active = min(active * rng.lognormal(0, 0.5), 0.9)
        score = np.where(rng.random(n_cells) < active, rng.beta(2, 2, n_cells), 0.0)
        coef[:, n_types + 2 + k] = lib * (1 - mito_frac) * strength * score

    doublet = np.zeros(n_cells, bool)
    if n_doublets:
        d = rng.choice(np.flatnonzero(~low_q), n_doublets, replace=False)
        partner = rng.choice(n_types, n_doublets, p=props)
        lib2 = rng.lognormal(np.log(median_umi * depth * u.type_lib[partner]), 0.5) * 0.9
        coef[d, partner] += lib2 * (1 - mito_frac[d])
        coef[d, n_types + 1] += lib2 * mito_frac[d]
        doublet[d] = True

    shape = u.dispersion_shape.astype(np.float32)
    blocks = []
    for start in range(0, n_cells, chunk):
        mu = coef[start:start + chunk] @ basis
        noise = rng.standard_gamma(shape, size=mu.shape).astype(np.float32) / shape
        counts = rng.poisson(mu * noise).astype(np.int32)
        blocks.append(sp.csr_matrix(counts))
    mat = sp.vstack(blocks, format="csr")

    # The cell caller drops barcodes with very few UMIs; mimic that.
    keep = np.asarray(mat.sum(1)).ravel() >= 500
    mat = mat[keep]
    truth = {
        "cell_type": np.array([CELL_TYPES[t][0] for t in ctype])[keep],
        "lineage": np.array([CELL_TYPES[t][1] for t in ctype])[keep],
        "is_doublet": doublet[keep],
        "is_low_quality": low_q[keep],
    }
    return mat, _barcodes(rng, mat.shape[0]), truth


def write_10x_h5(path: Path, mat: sp.csr_matrix, barcodes: np.ndarray, u: Universe,
                 library_id: str) -> None:
    """Write cells x genes CSR as CellRanger v3+ HDF5 (CSC genes x cells on disk)."""
    tmp = path.with_suffix(".tmp.h5")
    kw = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(tmp, "w") as f:
        f.attrs["filetype"] = "matrix"
        f.attrs["version"] = 2
        f.attrs["chemistry_description"] = "Single Cell 3' v3"
        f.attrs["software_version"] = "scbench synthetic"
        f.attrs["library_ids"] = np.array([library_id.encode()])
        f.attrs["original_gem_groups"] = np.array([1])
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=barcodes, **kw)
        g.create_dataset("data", data=mat.data.astype(np.int32), **kw)
        g.create_dataset("indices", data=mat.indices.astype(np.int64), **kw)
        g.create_dataset("indptr", data=mat.indptr.astype(np.int64), **kw)
        g.create_dataset("shape", data=np.array([mat.shape[1], mat.shape[0]], dtype=np.int32))
        feat = g.create_group("features")
        n = len(u.gene_ids)
        feat.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
        feat.create_dataset("feature_type", data=np.full(n, b"Gene Expression", dtype="S15"), **kw)
        feat.create_dataset("genome", data=np.full(n, GENOME.encode(), dtype="S6"), **kw)
        feat.create_dataset("id", data=u.gene_ids.astype("S15"), **kw)
        feat.create_dataset("name", data=u.gene_names.astype("S"), **kw)
    os.replace(tmp, path)


def _make_sample(args):
    out_dir, seed, idx, name, n_cells, batch = args
    u = build_universe(seed)
    mat, barcodes, truth = simulate_sample(u, n_cells, seed, idx, batch)
    sdir = Path(out_dir) / name
    sdir.mkdir(parents=True, exist_ok=True)
    write_10x_h5(sdir / "filtered_feature_bc_matrix.h5", mat, barcodes, u, name)
    with gzip.open(sdir / "truth.csv.gz", "wt") as fh:
        fh.write("barcode,cell_type,lineage,is_doublet,is_low_quality\n")
        for i, bc in enumerate(barcodes):
            fh.write(f"{bc.decode()},{truth['cell_type'][i]},{truth['lineage'][i]},"
                     f"{int(truth['is_doublet'][i])},{int(truth['is_low_quality'][i])}\n")
    nnz_per_cell = np.diff(mat.indptr)
    return {
        "name": name,
        "path": f"{name}/filtered_feature_bc_matrix.h5",
        "truth": f"{name}/truth.csv.gz",
        "n_cells": int(mat.shape[0]),
        "n_genes": int(mat.shape[1]),
        "nnz": int(mat.nnz),
        "median_umi": float(np.median(np.asarray(mat.sum(1)).ravel())),
        "median_genes": float(np.median(nnz_per_cell)),
    }


def dataset_dir(root: Path, size: str, kind: str, seed: int) -> Path:
    return root / f"{size}-seed{seed}" / kind


def generate(root: Path, size: str, kind: str, seed: int = 0, workers: int | None = None,
             log=print) -> Path:
    """Generate (or reuse) the dataset for ``kind`` in {"single", "atlas"}; returns its dir."""
    out = dataset_dir(root, size, kind, seed)
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("generator_version") == GENERATOR_VERSION:
        return out
    single_cells, n_atlas, atlas_mean = SIZES[size]
    if kind == "single":
        jobs = [(str(out), seed, 0, "sample01", single_cells, False)]
    else:
        rng = np.random.default_rng([seed, 2])
        cells = np.round(atlas_mean * rng.uniform(0.5, 1.5, n_atlas)).astype(int)
        jobs = [(str(out), seed, i + 1, f"S{i + 1:02d}", int(c), True) for i, c in enumerate(cells)]
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    log(f"generating {kind} dataset ({size}, {len(jobs)} sample(s)) in {out} ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        samples = list(ex.map(_make_sample, jobs))
    info = {
        "generator_version": GENERATOR_VERSION, "size": size, "kind": kind, "seed": seed,
        "n_cells_total": sum(s["n_cells"] for s in samples), "samples": samples,
    }
    info["fingerprint"] = hashlib.sha256(
        b"".join((out / s["path"]).read_bytes() for s in samples)).hexdigest()[:16]
    manifest.write_text(json.dumps(info, indent=2))
    log(f"  {info['n_cells_total']:,} cells, median UMI/cell "
        f"{np.median([s['median_umi'] for s in samples]):,.0f}, median genes/cell "
        f"{np.median([s['median_genes'] for s in samples]):,.0f}")
    return out
