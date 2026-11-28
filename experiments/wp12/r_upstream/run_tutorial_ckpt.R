## Restart-safe wrapper for the upstream tutorial (the machine rebooted three times on 2026-11-27).
## The tutorial code itself is still run VERBATIM and in order (tutorial_cells_verbatim.R, extracted
## mechanically from sir-hawkes-tutorial.ipynb). The harness only (a) sets a seed before the first cell,
## (b) saves the R workspace (which includes the RNG state .Random.seed) after each notebook cell, and
## (c) on restart reloads the last workspace, re-attaches the packages and continues with the next cell.
## usage: Rscript run_tutorial_ckpt.R <upstream_repo_dir> <cells_file> <out_dir> <seed>
args <- commandArgs(trailingOnly = TRUE)
upstream <- args[1]; cells <- normalizePath(args[2]); outdir <- normalizePath(args[3]); seed <- as.integer(args[4])
setwd(upstream)
lines <- readLines(cells)
starts <- grep("^## ---- notebook cell", lines)
ends <- c(starts[-1] - 1, length(lines))
ck <- function(k) file.path(outdir, sprintf("ckpt_cell_%02d.RData", k))
done <- which(file.exists(sapply(seq_along(starts), ck)))
first <- 1
if (length(done) > 0) {
  first <- max(done) + 1
  .keep <- c(first = first)
  load(ck(max(done)), envir = globalenv())
  first <- .keep["first"]
  suppressMessages({library(parallel); library(nloptr); library(data.table); library(Matrix)})
  cat(sprintf("[harness] resumed after cell block %d of %d\n", first - 1, length(starts)))
} else {
  set.seed(seed)
}
cat("R:", R.version.string, " nloptr", as.character(packageVersion("nloptr")),
    " data.table", as.character(packageVersion("data.table")),
    " Matrix", as.character(packageVersion("Matrix")), " seed", seed, "\n")
for (k in seq_along(starts)) {
  if (k < first) next
  seg <- lines[starts[k]:ends[k]]
  is_plot <- any(grepl("^matplot", seg))
  if (is_plot) png(file.path(outdir, "tutorial_size_distribution_R.png"), width = 900, height = 700)
  t0 <- Sys.time()
  source(textConnection(seg), echo = TRUE, max.deparse.length = Inf)
  if (is_plot) dev.off()
  cat(sprintf("\n[harness] cell block %d (%s) took %s\n", k, lines[starts[k]], format(Sys.time() - t0)))
  .harness <- list(k = k)
  save(list = setdiff(ls(globalenv(), all.names = TRUE), c("args", "lines", "starts", "ends", "ck", "done", "first", "k", "seg", "is_plot", "t0", "upstream", "cells", "outdir", "seed")),
       file = ck(k), envir = globalenv())
}
## ---- exports (after the tutorial; reads its variables, changes nothing) ----
for (i in 1:nsim) {
  write.csv(as.data.frame(simdat[, i]), file.path(outdir, sprintf("sim_state_%02d.csv", i)), row.names = FALSE)
  write.csv(as.data.frame(simhistory[, i]), file.path(outdir, sprintf("sim_history_%02d.csv", i)), row.names = FALSE)
}
write.csv(complete_res, file.path(outdir, "fit_sir.csv"))
write.csv(res, file.path(outdir, "fit_hawkesn.csv"))
write.csv(data.frame(size = as.numeric(names(size.est.at.zero$final.state)),
                     prior = as.numeric(size.est.at.zero$final.state),
                     posterior = as.numeric(size.est.at.end$final.state)),
          file.path(outdir, "size_distribution_R.csv"), row.names = FALSE)
writeLines(c(sprintf("seenEvents=%d", seenEvents), sprintf("n_history=%d", nrow(history)),
             sprintf("theo_mean_prior=%.6f", size.est.at.zero$theo.mean),
             sprintf("theo_mean_post=%.6f", size.est.at.end$theo.mean)),
           file.path(outdir, "size_distribution_R_meta.txt"))
cat("[harness] exports written\n")
