# Relaunch after the second machine restart. Run from the repo root. Logs: experiments/WP-02/logs-restart2/
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
L=experiments/WP-02/logs-restart2
uv run python experiments/wp02/run_e1_window_counts.py --proposal v2 --n-cloud 120000 --only-cells 81 --shard 0 --n-shards 1 > $L/e1-cell81.txt 2>&1 &
WP02_BOX_LOGIT_I0_LO=-14 uv run python experiments/wp02/run_e1_v3.py 10 22 > $L/box-14.txt 2>&1 &
WP02_BOX_LOGIT_I0_LO=-20 uv run python experiments/wp02/run_e1_v3.py 10 22 > $L/box-20.txt 2>&1 &
uv run python experiments/wp02/run_e2_e3.py --experiment e3 --only-cells 8,9 --shard 0 --n-shards 2 > $L/e3-s0.txt 2>&1 &
uv run python experiments/wp02/run_e2_e3.py --experiment e3 --only-cells 8,9 --shard 1 --n-shards 2 > $L/e3-s1.txt 2>&1 &
uv run python experiments/wp02/run_e2_e3.py --experiment e2 --only-cells 0,1,2,3,4,5,6,7,8,9,10,11 --shard 0 --n-shards 1 > $L/e2-s0.txt 2>&1 &
wait
echo RESTART2DONE > $L/DONE
