# queue: box -20 after refit check; E2 after E1 completes. Run from repo root.
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
( until [ -f experiments/WP-02/e1-refit-check/cell-010.json ]; do sleep 15; done
  WP02_BOX_LOGIT_I0_LO=-20 uv run python experiments/wp02/run_e1_v3.py 53 4 10 22 > experiments/WP-02/e1-prior-box-check/log-20.txt 2>&1 ) &
( until [ $(ls experiments/WP-02/e1-window-counts/cell-*.npz | wc -l) -ge 84 ]; do sleep 15; done
  mkdir -p experiments/WP-02/e2-bins
  for s in 0 1; do uv run python experiments/wp02/run_e2_e3.py --experiment e2 --shard $s --n-shards 2 > experiments/WP-02/e2-bins/log-shard$s.txt 2>&1 & done; wait ) &
wait
echo QUEUEDONE
