"""Check calib.derived() closed forms (peak time, peak height, final size) against a dense numerical ODE solve."""
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[2]))
import jax.numpy as jnp
from experiments.wp02 import calib as C
from experiments.wp02.sir import solve_sir

rows = []
for beta, gamma, i0 in [(0.6, .2, .003), (1, .67, .003), (.5, .05, .003), (.3, .35, .003), (1, 1, .003), (3, .1, .01), (0.2, 0.1, 1e-4)]:
    th = jnp.array([np.log(beta), np.log(gamma), 0.0, np.log(i0 / (1 - i0))])
    ts = np.linspace(0, 3000, 300001)
    ys = np.asarray(solve_sir(th, jnp.asarray(ts)))
    I = ys[:, 1]; cum = ys[:, 3]
    d = C.derived(np.log(beta), np.log(gamma), np.log(i0 / (1 - i0)), 0.0)
    T = C.TARGETS
    rows.append((beta, gamma, i0, ts[np.argmax(I)], d[T.index("peak_time")], np.log(I.max()), d[T.index("log_peak_rate")], np.log(cum[-1]), d[T.index("log_total")]))
    print("b=%.2f g=%.2f i0=%g | tpeak ode %.3f formula %.3f | logpeak ode %.5f f %.5f | logtotal ode %.5f f %.5f" % rows[-1])
np.savetxt("experiments/WP-02/e1-checks/derived_vs_ode.txt", np.array(rows), header="beta gamma i0 tpeak_ode tpeak_formula logpeak_ode logpeak_formula logtotal_ode logtotal_formula")
