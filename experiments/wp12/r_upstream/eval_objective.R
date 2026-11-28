## Evaluates the upstream HawkesN objective (neg.log.likelihood) and, optionally, runs the upstream
## fitter (fitSeries) on cascades given as CSV. Upstream functions are sourced unmodified.
## usage: Rscript eval_objective.R <upstream_repo_dir> <cascade_dir> <points_csv|NONE> <out_csv> <mode: eval|fit> [ncores]
args <- commandArgs(trailingOnly = TRUE)
upstream <- args[1]; cdir <- normalizePath(args[2]); pts <- args[3]; outcsv <- args[4]; mode <- args[5]
ncores <- if (length(args) >= 6) as.integer(args[6]) else 1
if (pts != "NONE") pts <- normalizePath(pts)
outcsv <- file.path(normalizePath(dirname(outcsv)), basename(outcsv))
setwd(upstream)
suppressMessages({library(parallel); source('scripts/functions-SIR-HawkesN.R')})
read_hist <- function(id) {
  d <- read.csv(file.path(cdir, sprintf("%s.csv", id)))
  data.frame(magnitude = 1.1, time = d$time)   # dummy magnitude 1.1, as the tutorial's own converter does
}
if (mode == "eval") {
  P <- read.csv(pts, stringsAsFactors = FALSE)
  P$upstream_nll <- sapply(seq_len(nrow(P)), function(r) {
    h <- read_hist(P$id[r])
    suppressWarnings(neg.log.likelihood(c(K = P$K[r], c = 0.1, theta = P$theta[r], N = P$N[r]), h))
  })
  write.csv(P, outcsv, row.names = FALSE)
} else {
  ids <- sub("\\.csv$", "", list.files(cdir, pattern = "\\.csv$"))
  starts <- list(tutorial = c(K = 1, c = 0.1, theta = 0.1, N = 1000))
  if (pts != "NONE") {   # extra start points per cascade (columns id, K, theta, N, label)
    S <- read.csv(pts, stringsAsFactors = FALSE)
  } else S <- NULL
  jobs <- list()
  for (id in ids) {
    jobs[[length(jobs) + 1]] <- list(id = id, label = "tutorial_start", x0 = starts$tutorial)
    if (!is.null(S)) for (r in which(S$id == id))
      jobs[[length(jobs) + 1]] <- list(id = id, label = S$label[r], x0 = c(K = S$K[r], c = 0.1, theta = S$theta[r], N = S$N[r]))
  }
  out <- mclapply(jobs, function(j) {
    h <- read_hist(j$id)
    t0 <- Sys.time()
    m <- suppressWarnings(fitSeries(history = h, j$x0))
    data.frame(id = j$id, label = j$label, n = nrow(h), K = m$par["K"], theta = m$par["theta"], N = m$par["N"],
               nll = m$value, iter = ifelse(is.null(m$iter), NA, m$iter),
               conv = ifelse(is.null(m$convergence), NA, m$convergence),
               secs = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  }, mc.cores = ncores)
  write.csv(do.call(rbind, out), outcsv, row.names = FALSE)
}
