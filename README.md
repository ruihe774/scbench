# scbench

A CPU benchmark for single-cell RNA-seq analysis. It runs the standard
analysis pipelines people actually use, with the popular tools, on synthetic
10x Chromium data, and reports wall time, CPU use and memory for every step.

|                   | **single** (one 10x sample)      | **atlas** (many samples + integration)            |
|-------------------|----------------------------------|---------------------------------------------------|
| **scanpy**        | `scanpy-single`                  | `scanpy-atlas`                                    |
| **Seurat v5**     | `seurat-single`                  | `seurat-atlas`                                    |

Everything runs on CPU; no GPU libraries are installed.

## Quick start

Requires [pixi](https://pixi.sh). The first run installs the conda environments,
which can take a few minutes.

```bash
pixi run smoke                # tiny dataset, all 4 variants, ~2-3 min: checks the setup
pixi run bench                # standard size, all 4 variants
pixi run bench --size large --repeats 3
pixi run bench --variants scanpy-single,seurat-single --threads 8
pixi run report results/<run-dir>
pixi run compare results/<run-a> results/<run-b> --steps   # e.g. two machines
```

`pixi run bench --help` lists all options. Results go to
`results/<timestamp>-<host>-<size>/`:

- `results.json`: system info, configuration, per-run and median per-step metrics
- `<variant>/rep<N>/events.jsonl`: step start/end times
- `<variant>/rep<N>/timeline.csv`: memory and CPU-time samples every 0.25 s
- `<variant>/rep<N>/workload.log`: stdout/stderr of the workload
- `<variant>/rep<N>/info.json`: package versions, BLAS, cell/cluster counts
- `<variant>/rep<N>/clusters.csv.gz`: final cluster assignment (used for validation)

## Dataset sizes

| size       | single sample | atlas                                    |
|------------|---------------|------------------------------------------|
| `tiny`     | 2,000 cells   | 4 samples, ~6.5k cells (smoke test)      |
| `standard` | 10,000 cells  | 12 samples, ~82k cells                   |
| `large`    | 20,000 cells  | 32 samples, ~320k cells                  |

Datasets are generated deterministically from `--seed` (default 0) on first use,
then cached in `data/`. Each dataset has a content fingerprint that is stored in
the results, so you can check that two machines ran the same input.

## Synthetic data

`scbench/generate.py` writes CellRanger-format `filtered_feature_bc_matrix.h5`
files: HDF5 v3 layout, gzip-compressed CSC matrix, `int32` counts, 16-bp + `-1`
barcodes, and `features/{id,name,feature_type,genome}`. Both
`scanpy.read_10x_h5` and `Seurat::Read10X_h5` read them unchanged. The data is
built to look like a 10x 3' v3 PBMC sample:

- **Features:** 36,601 genes, the size of GRCh38-2020-A. Mitochondrial,
  ribosomal, hemoglobin, sex-linked and housekeeping genes have their real
  symbols, so the usual `^MT-` / `^RP[SL]` / `^HB` QC patterns work. lncRNA-style
  names and a few duplicated symbols exercise the make-unique code paths.
- **Expression:** heavy-tailed gene abundances. 16 PBMC-like cell types in a
  lineage → subtype hierarchy carry lineage and subtype marker genes. Shared
  continuous programmes (cell cycle, interferon, stress, activation) add
  heterogeneity within clusters.
- **Counts:** gamma-Poisson (negative binomial) with gene-specific overdispersion
  and log-normal library sizes. Median ≈ 6.5k UMIs and ≈ 2.3k genes per cell,
  ~93% zeros, ~28k genes detected.
- **Artefacts:** ~5% mitochondrial reads, ~22% ribosomal, 5% low-quality
  high-mito cells, ambient RNA contamination, and heterotypic doublets at the
  10x rate (0.8% per 1,000 cells). Barcodes under 500 UMIs are dropped, as
  CellRanger's cell calling would.
- **Atlas samples:** each sample also gets donor sex (XIST vs chrY genes),
  shifted cell type composition, a different sequencing depth, gene-wise batch
  effects, and different programme prevalence.

Each sample comes with ground truth (`truth.csv.gz`). After every run, scbench
computes the ARI of the clustering against true cell types and lineages, and
the fraction of simulated doublets removed. These numbers catch broken runs;
they are not a quality score.

## Pipelines

Both frameworks run the same canonical steps with their own idiomatic defaults.
Atlas variants merge all samples, use per-sample doublet calling and HVG
selection, and integrate with Harmony.

| step                | scanpy                                                   | Seurat v5                                               |
|---------------------|----------------------------------------------------------|---------------------------------------------------------|
| `load_libraries`    | import scanpy & co.                                      | `library(Seurat)` & co.                                 |
| `read_data`         | `read_10x_h5` (+ `anndata.concat`)                       | `Read10X_h5`, `CreateSeuratObject(min.cells=3, min.features=200)` (+ `merge`) |
| `qc_metrics`        | `calculate_qc_metrics` (mt / ribo / hb)                  | `PercentageFeatureSet` (mt / ribo / hb)                 |
| `qc_filter`         | ≥200 genes, genes in ≥3 cells, mito < 20%                | ≥200 genes, mito < 20%                                  |
| `doublet_detection` | `pp.scrublet` (per sample)                               | `scDblFinder` (per sample, BiocParallel)                |
| `normalize`         | `normalize_total(1e4)` + `log1p`                         | (`split` layers) `NormalizeData`                        |
| `hvg`               | `highly_variable_genes(2000)`, Seurat flavour            | `FindVariableFeatures(vst, 2000)`                       |
| `scale`             | `scale(max_value=10)` on HVGs                            | `ScaleData` on HVGs                                     |
| `pca`               | `pca(50)`                                                | `RunPCA(50)`                                            |
| `integration`       | Harmony (`harmonypy`) - atlas only                       | `IntegrateLayers(HarmonyIntegration)` - atlas only      |
| `neighbors`         | `neighbors(15, 30 PCs)` (pynndescent/umap)               | `FindNeighbors(30 dims)` (Annoy, SNN)                   |
| `clustering`        | `leiden(1.0, igraph)`                                    | `FindClusters(0.8)` (Louvain)                           |
| `umap`              | `tl.umap`                                                | `RunUMAP` (uwot)                                        |
| `markers`           | `rank_genes_groups(wilcoxon)`                            | `FindAllMarkers` (Wilcoxon via presto)                  |
| `save`              | `write_h5ad`                                             | `saveRDS`                                               |

The software stack is pinned by `pixi.lock`: conda-forge and bioconda, Python
3.12, R 4.4, OpenBLAS. Its hash is stored with every result.

## What gets measured

- **Step wall time** comes from the workload's own timestamps around each
  step. **Total** is the sum of the steps, including library loading, which
  users pay too.
- **Peak memory** sums proportional set size (PSS) over the whole process
  tree on Linux, so forked BiocParallel workers count correctly. On macOS it
  sums RSS. The tree is sampled every 0.25 s.
- **Cores** is the average number of busy cores during a step, taken from
  CPU time across the process tree. It shows which steps actually run in
  parallel.
- **Threads:** `--threads` (default: physical cores) sets
  `OMP/OPENBLAS/MKL/NUMBA/RCPP_PARALLEL_NUM_THREADS`, `scanpy.settings.n_jobs`,
  harmonypy `ncores`, and the BiocParallel worker count (capped at 4 for
  scDblFinder, see below). Everything else runs with the tool's defaults.
- **scDblFinder workers are capped at 4.** Each forked worker holds its own
  working set of about 7 GB, so one worker per sample peaked at 85 GB on the
  standard atlas, only 13 s faster than 4 workers (62 s vs 75 s, 38 GB).
- **Seurat's `markers` step is slow by design, and single-threaded.**
  `FindAllMarkers` calls `FindMarkers` once per cluster, and each call ranks
  the whole matrix again, so cost grows with clusters x cells. One presto call
  over all clusters would be about 60x faster on the standard atlas. The
  benchmark keeps `FindAllMarkers` because it is what Seurat users run. It
  dominates seurat-atlas wall time (about 60%), and `saveRDS` adds more
  single-core time.
- With `--repeats N`, the summary reports per-step medians. Numba JIT
  compilation is cached after the first run, so for scanpy the first repeat
  can be slower.

For comparable numbers, run on an idle machine with the same `--size` and
`--threads`. Use the `performance` CPU governor on Linux if you can. Results
record the governor and the load average at start.

## Layout

```
pixi.toml                    environments (default / scanpy / seurat) and tasks
scbench/generate.py          synthetic 10x data generator
scbench/cli.py               run / report / compare / generate
scbench/monitor.py           process-tree CPU & memory sampler
scbench/sysinfo.py           machine description
workloads/scanpy_workload.py
workloads/seurat_workload.R
```

Bioconda's `GenomeInfoDbData`, a dependency of scDblFinder, normally downloads
its data in a post-link script, which pixi does not run. scbench runs that
script once, automatically, the first time it prepares the `seurat`
environment.
