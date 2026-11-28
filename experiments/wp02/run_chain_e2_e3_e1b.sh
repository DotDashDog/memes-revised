cd /home/linus/memes-revised
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f run_e1_window_counts.py >/dev/null; do sleep 10; done
mkdir -p experiments/WP-02/e3-noise-spikes experiments/WP-02/e2-bins experiments/WP-02/e1-refit-check
for s in 0 1 2; do uv run python experiments/wp02/run_e2_e3.py --experiment e3 --shard $s --n-shards 3 > experiments/WP-02/e3-noise-spikes/log-shard$s.txt 2>&1 & done
for s in 0 1; do uv run python experiments/wp02/run_e2_e3.py --experiment e2 --shard $s --n-shards 2 > experiments/WP-02/e2-bins/log-shard$s.txt 2>&1 & done
(uv run python experiments/wp02/run_refit_check.py 5 > experiments/WP-02/e1-refit-check/log-5.txt 2>&1; uv run python experiments/wp02/run_refit_check.py 10 > experiments/WP-02/e1-refit-check/log-10.txt 2>&1; echo done > experiments/WP-02/e1-refit-check/DONE) &
wait
echo ALLDONE
