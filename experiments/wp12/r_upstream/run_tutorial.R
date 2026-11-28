## Wrapper that runs the upstream tutorial's code cells VERBATIM (tutorial_cells_verbatim.R,
## extracted mechanically from sir-hawkes-tutorial.ipynb) and then exports what it produced.
## Nothing inside the tutorial code is changed. The only additions are outside it:
## a fixed RNG seed (the tutorial sets none), a PNG device for its plot, and CSV exports.
## usage: Rscript run_tutorial.R <upstream_repo_dir> <cells_file> <out_dir> <seed>
args <- commandArgs(trailingOnly = TRUE)
upstream <- args[1]; cells <- normalizePath(args[2]); outdir <- normalizePath(args[3]); seed <- as.integer(args[4])
setwd(upstream)
set.seed(seed)
cat("R:", R.version.string, " nloptr", as.character(packageVersion("nloptr")),
    " data.table", as.character(packageVersion("data.table")),
    " Matrix", as.character(packageVersion("Matrix")), " seed", seed, "\n")
png(file.path(outdir, "tutorial_size_distribution_R.png"), width = 900, height = 700)
t0 <- Sys.time()
source(cells, echo = TRUE, max.deparse.length = Inf)
dev.off()
cat("\n[wrapper] tutorial code finished in", format(Sys.time() - t0), "\n")
## ---- exports (after the tutorial; reads its variables, changes nothing) ----
for (i in 1:nsim) {
  write.csv(as.data.frame(simdat[, i]), file.path(outdir, sprintf("sim_state_%02d.csv", i)), row.names = FALSE)
  write.csv(as.data.frame(simhistory[, i]), file.path(outdir, sprintf("sim_history_%02d.csv", i)), row.names = FALSE)
}
write.csv(complete_res, file.path(outdir, "fit_sir.csv"))
write.csv(res, file.path(outdir, "fit_hawkesn.csv"))
hv <- data.frame(value = unlist(results["value", ]))
write.csv(hv, file.path(outdir, "fit_hawkesn_nll.csv"))
write.csv(data.frame(size = as.numeric(names(size.est.at.zero$final.state)),
                     prior = as.numeric(size.est.at.zero$final.state),
                     posterior = as.numeric(size.est.at.end$final.state)),
          file.path(outdir, "size_distribution_R.csv"), row.names = FALSE)
writeLines(c(sprintf("seenEvents=%d", seenEvents), sprintf("n_history=%d", nrow(history)),
             sprintf("theo_mean_prior=%.6f", size.est.at.zero$theo.mean),
             sprintf("theo_mean_post=%.6f", size.est.at.end$theo.mean)),
           file.path(outdir, "size_distribution_R_meta.txt"))
cat("[wrapper] exports written\n")
