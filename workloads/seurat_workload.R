# scbench Seurat workload: a standard Seurat v5 analysis of one 10x sample or an atlas.
#
# Each timed step appends {"step", "start", "end"} (epoch seconds) to the events
# file so the runner can attribute wall time, CPU use and memory to steps.

T0 <- as.numeric(Sys.time())

parse_args <- function(argv) {
  keys <- sub("^--", "", argv[c(TRUE, FALSE)])
  setNames(as.list(argv[c(FALSE, TRUE)]), keys)
}
args <- parse_args(commandArgs(trailingOnly = TRUE))
threads <- as.integer(args$threads)
atlas <- args$mode == "atlas"
events <- file(args$events, open = "a")

# `expr` is a promise, so it is evaluated (and assigns) in the caller's environment.
step <- function(name, expr, start = NULL) {
  t0 <- if (is.null(start)) as.numeric(Sys.time()) else start
  force(expr)
  writeLines(sprintf('{"step": "%s", "start": %.6f, "end": %.6f}', name, t0, as.numeric(Sys.time())),
             events)
  flush(events)
  invisible(NULL)
}

step("load_libraries", start = T0, {
  suppressPackageStartupMessages({
    library(Seurat)
    library(scDblFinder)
    library(harmony)
    library(BiocParallel)
    library(jsonlite)
  })
  options(future.globals.maxSize = Inf)
})

manifest <- fromJSON(file.path(args$data, "manifest.json"))
samples <- manifest$samples
# Each forked scDblFinder worker holds its own copy of a sample's working set (~7 GB for 7k cells), and
# 12 workers peaked at 85 GB on the standard atlas for only 13 s saved over 4 workers.
MAX_DBL_WORKERS <- 4
bpparam <- if (atlas && threads > 1) MulticoreParam(workers = min(threads, nrow(samples), MAX_DBL_WORKERS)) else SerialParam()
set.seed(0)

step("read_data", {
  objs <- lapply(seq_len(nrow(samples)), function(i) {
    counts <- Read10X_h5(file.path(args$data, samples$path[i]))
    o <- CreateSeuratObject(counts, project = samples$name[i], min.cells = 3, min.features = 200)
    o$sample <- samples$name[i]
    o$barcode <- colnames(o)
    o
  })
  if (atlas) {
    obj <- merge(objs[[1]], objs[-1], add.cell.ids = samples$name)
    obj <- JoinLayers(obj)
  } else {
    obj <- objs[[1]]
  }
  rm(objs)
})
n_input <- sum(samples$n_cells)

step("qc_metrics", {
  obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
  obj[["percent.ribo"]] <- PercentageFeatureSet(obj, pattern = "^RP[SL]")
  obj[["percent.hb"]] <- PercentageFeatureSet(obj, pattern = "^HB[^(P)]")
})

step("qc_filter", {
  obj <- subset(obj, subset = nFeature_RNA >= 200 & percent.mt < 20)
})

step("doublet_detection", {
  sce <- scDblFinder(LayerData(obj, layer = "counts"), samples = if (atlas) obj$sample else NULL,
                     BPPARAM = bpparam)
  is_doublet <- sce$scDblFinder.class == "doublet"
  n_doublets <- sum(is_doublet)
  obj <- obj[, !is_doublet]
  rm(sce)
})

step("normalize", {
  if (atlas) obj[["RNA"]] <- split(obj[["RNA"]], f = obj$sample)
  obj <- NormalizeData(obj, verbose = FALSE)
})

step("hvg", {
  obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
})

step("scale", {
  obj <- ScaleData(obj, verbose = FALSE)
})

step("pca", {
  obj <- RunPCA(obj, npcs = 50, verbose = FALSE)
})

reduction <- "pca"
if (atlas) {
  step("integration", {
    obj <- IntegrateLayers(obj, method = HarmonyIntegration, orig.reduction = "pca",
                           new.reduction = "harmony", verbose = FALSE)
    reduction <- "harmony"
  })
}

step("neighbors", {
  obj <- FindNeighbors(obj, reduction = reduction, dims = 1:30, verbose = FALSE)
})

step("clustering", {
  obj <- FindClusters(obj, resolution = 0.8, random.seed = 0, verbose = FALSE)
})

step("umap", {
  obj <- RunUMAP(obj, reduction = reduction, dims = 1:30, seed.use = 0, verbose = FALSE)
})

step("markers", {
  if (atlas) obj <- JoinLayers(obj)
  markers <- FindAllMarkers(obj, verbose = FALSE)
})

step("save", {
  saveRDS(obj, file.path(args$outdir, "result.rds"))
})
close(events)

gz <- gzfile(file.path(args$outdir, "clusters.csv.gz"), "w")
write.csv(data.frame(sample = obj$sample, barcode = obj$barcode, cluster = obj$seurat_clusters),
          gz, row.names = FALSE, quote = FALSE)
close(gz)

pkgs <- c("Seurat", "SeuratObject", "Matrix", "harmony", "scDblFinder", "presto", "uwot",
          "RcppAnnoy", "irlba", "hdf5r", "xgboost", "BiocParallel")
versions <- c(list(R = paste(R.version$major, R.version$minor, sep = ".")),
              setNames(lapply(pkgs, function(p) tryCatch(as.character(packageVersion(p)),
                                                         error = function(e) NULL)), pkgs))
info <- list(
  framework = "seurat",
  versions = versions,
  blas = c(extSoftVersion()[["BLAS"]], La_library()),
  metrics = list(
    n_cells_input = n_input,
    n_doublets_removed = n_doublets,
    n_cells_final = ncol(obj),
    n_genes_final = nrow(obj),
    n_hvg = length(VariableFeatures(obj)),
    n_clusters = length(levels(obj$seurat_clusters)),
    n_markers = nrow(markers)
  )
)
writeLines(toJSON(info, auto_unbox = TRUE, pretty = TRUE, null = "null"),
           file.path(args$outdir, "info.json"))
