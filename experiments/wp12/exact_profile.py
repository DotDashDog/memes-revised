"""Optimiser-free HawkesN likelihood tools (mu = 0), independent of hawkesn.py.

Model (Rizoiu et al. 2018, exponential kernel, no background rate):

    lambda(t) = (1 - N_t/N) * K*theta * sum_{t_s < t} exp(-theta (t - t_s))

with I0 seed events at t = 0 that count towards N_t and the kernel sum but are not
themselves "observed" events.

Key fact used throughout.  Write intervals k = 0..n between consecutive times
tau_0 = 0, tau_k = t_k, tau_{n+1} = T.  During interval k exactly m_k = I0 + k
individuals are used up.  With A_k^+ the kernel sum just after tau_k,

    c_k(theta)  = A_k^+ (1 - exp(-theta * Delta_k))          (kernel mass in interval k)
    C0(theta)   = sum_k c_k ,   C1(theta) = sum_k m_k c_k
    S_A(theta)  = sum_{i=1..n} log A_i^-                     (kernel sum just before event i)
    D(N)        = sum_{i=1..n} log(1 - (I0 + i - 1)/N)

    loglik(K, theta, N) = n log K + n log theta + S_A(theta) + D(N) - K (C0 - C1/N)

So the likelihood depends on the data only through three functions of theta and one
of N, and for fixed (theta, N) the maximising K is  K_hat = n / (C0 - C1/N).
Everything below evaluates this on dense grids; there is no numerical optimiser,
so "the optimiser missed the peak" cannot happen here.

Assumptions (surfaced per method-discipline section 7): I0 known; mu = 0; all events
observed on [0, T]; N treated as a continuous parameter >= I0 + n.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

CHI2_1_95 = 1.9207294103470749  # 0.5 * chi2.ppf(0.95, 1)


@dataclasses.dataclass
class SuffStats:
    thetas: np.ndarray  # (nth,)
    S_A: np.ndarray  # (nth,)
    C0: np.ndarray  # (nth,)
    C1: np.ndarray  # (nth,)
    n: int
    I0: int
    T: float


def suff_stats(times: np.ndarray, T: float, I0: int, thetas: np.ndarray) -> SuffStats:
    """O(n * len(thetas)) forward recursion."""
    times = np.asarray(times, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    n = len(times)
    A = np.full(thetas.shape, float(I0))  # kernel sum just after tau_0 = 0
    S_A = np.zeros_like(thetas)
    C0 = np.zeros_like(thetas)
    C1 = np.zeros_like(thetas)
    t_prev = 0.0
    for k in range(n + 1):
        t_next = times[k] if k < n else T
        decay = np.exp(-thetas * (t_next - t_prev))
        c = A * (1.0 - decay)
        C0 += c
        C1 += (I0 + k) * c
        if k < n:
            A_minus = A * decay
            S_A += np.log(np.maximum(A_minus, 1e-300))
            A = A_minus + 1.0
        t_prev = t_next
    return SuffStats(thetas=thetas, S_A=S_A, C0=C0, C1=C1, n=n, I0=I0, T=float(T))


def D_of_N(Ns: np.ndarray, n: int, I0: int) -> np.ndarray:
    """sum_{i=1..n} log(1 - (I0+i-1)/N) for each N; -inf where N <= I0+n-1."""
    Ns = np.asarray(Ns, dtype=np.float64)
    m = I0 + np.arange(n, dtype=np.float64)  # I0 .. I0+n-1
    out = np.full(Ns.shape, -np.inf)
    ok = Ns > (I0 + n - 1)
    if n == 0:
        return np.where(ok, 0.0, -np.inf)
    out[ok] = np.log1p(-m[None, :] / Ns[ok][:, None]).sum(axis=1)
    return out


def loglik_point(times, T, I0, K, theta, N) -> float:
    """Log-likelihood at one parameter point through the sufficient statistics."""
    ss = suff_stats(times, T, I0, np.array([theta]))
    D = D_of_N(np.array([N]), ss.n, I0)[0]
    return float(
        ss.n * math.log(K) + ss.n * math.log(theta) + ss.S_A[0] + D - K * (ss.C0[0] - ss.C1[0] / N)
    )


def default_N_grid(n_total: int, n_pts: int = 400, max_mult: float = 1000.0) -> np.ndarray:
    """N = n_total + {0} U exp(u): starts AT the observed count n_total = I0 + n, dense
    just above it, log-spaced up to max_mult * n_total.

    N < n_total is excluded: there the depletion factor after the last event,
    1 - n_total/N, is negative and the linear form C0 - C1/N would subtract a
    negative compensator (a spuriously high likelihood).  hawkesn.py clips that
    factor at zero instead; under clipping N in (n_total-1, n_total) is a continuous
    extension with no modelling meaning (the population cannot be smaller than the
    number of posts already seen), so the grid simply starts at N = n_total."""
    u = np.linspace(math.log(1e-3), math.log(max_mult * n_total), n_pts - 1)
    return n_total + np.concatenate([[0.0], np.exp(u)])


@dataclasses.dataclass
class GridFit:
    """Everything needed for profiles of one cascade."""

    ss: SuffStats
    Ns: np.ndarray
    D: np.ndarray
    ll_thN: np.ndarray  # (nth, nN): log-lik maximised over K analytically
    K_hat_thN: np.ndarray  # (nth, nN)
    ll_max: float
    mle: dict


def grid_fit(times, T, I0, thetas, Ns=None) -> GridFit:
    ss = suff_stats(times, T, I0, thetas)
    n = ss.n
    if Ns is None:
        Ns = default_N_grid(I0 + n)
    Ns = np.asarray(Ns, dtype=np.float64)
    Ns = Ns[Ns >= I0 + n]  # see default_N_grid: N below the observed count is excluded
    D = D_of_N(Ns, n, I0)
    denom = ss.C0[:, None] - ss.C1[:, None] / Ns[None, :]
    denom = np.where(denom > 0, denom, np.nan)
    K_hat = n / denom
    ll = n * np.log(K_hat) - n + (n * np.log(ss.thetas) + ss.S_A)[:, None] + D[None, :]
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    mle = dict(
        K=float(K_hat[i, j]),
        theta=float(ss.thetas[i]),
        N=float(Ns[j]),
        i_theta=int(i),
        j_N=int(j),
        theta_at_edge=bool(i in (0, len(thetas) - 1)),
        N_at_edge=bool(j in (0, len(Ns) - 1)),
    )
    return GridFit(ss=ss, Ns=Ns, D=D, ll_thN=ll, K_hat_thN=K_hat, ll_max=float(ll[i, j]), mle=mle)


def ll_given_K_theta(gf: GridFit, K_of_theta: np.ndarray) -> np.ndarray:
    """max over N of loglik(K_of_theta[i], theta_i, N) -> (nth,).

    K_of_theta may contain nan/non-positive entries (infeasible) -> -inf."""
    ss = gf.ss
    n = ss.n
    K = np.asarray(K_of_theta, dtype=np.float64)
    ok = np.isfinite(K) & (K > 0)
    Ks = np.where(ok, K, 1.0)
    base = n * np.log(Ks) + n * np.log(ss.thetas) + ss.S_A - Ks * ss.C0  # (nth,)
    full = base[:, None] + gf.D[None, :] + (Ks * ss.C1)[:, None] / gf.Ns[None, :]
    out = full.max(axis=1)
    return np.where(ok, out, -np.inf)


def profile_generic(gf: GridFit, values: np.ndarray, K_from) -> np.ndarray:
    """Profile log-lik of a derived quantity v, where K = K_from(v, theta).

    e.g. beta = K*theta      -> K_from = lambda v, th: v / th
         r = theta*(K-1)     -> K_from = lambda v, th: 1 + v / th
         K itself            -> K_from = lambda v, th: v + 0*th
    Maximises over theta (grid) and N (grid)."""
    out = np.empty(len(values))
    for a, v in enumerate(values):
        out[a] = ll_given_K_theta(gf, K_from(v, gf.ss.thetas)).max()
    return out


def profile_theta(gf: GridFit) -> np.ndarray:
    return gf.ll_thN.max(axis=1)


def profile_N(gf: GridFit) -> np.ndarray:
    return gf.ll_thN.max(axis=0)


def surface_K_theta(gf: GridFit, Ks: np.ndarray) -> np.ndarray:
    """(nK, nth) profile surface: max over N of loglik(K, theta, N)."""
    out = np.empty((len(Ks), len(gf.ss.thetas)))
    for a, K in enumerate(Ks):
        out[a] = ll_given_K_theta(gf, np.full(len(gf.ss.thetas), K))
    return out


def interval_from_profile(x: np.ndarray, prof: np.ndarray, ll_max: float | None = None,
                          cut: float = CHI2_1_95) -> dict:
    """Outermost crossings of prof = max - cut, linearly interpolated on the grid.

    Returns lo, hi, and flags lo_open/hi_open when the profile is still above the
    cut at the grid edge (interval not closed inside the grid box).  `ll_max`
    defaults to the profile's own maximum."""
    x = np.asarray(x, dtype=np.float64)
    prof = np.asarray(prof, dtype=np.float64)
    m = np.nanmax(prof) if ll_max is None else ll_max
    above = prof >= (m - cut)
    if not above.any():
        k = int(np.nanargmax(prof))
        return dict(lo=float(x[k]), hi=float(x[k]), lo_open=False, hi_open=False, mode=float(x[k]))
    idx = np.where(above)[0]
    a, b = idx[0], idx[-1]

    def cross(i_in, i_out):
        y0, y1 = prof[i_in] - (m - cut), prof[i_out] - (m - cut)
        if not np.isfinite(y1):
            return float(x[i_in])
        f = y0 / (y0 - y1)
        return float(x[i_in] + f * (x[i_out] - x[i_in]))

    lo_open = a == 0
    hi_open = b == len(x) - 1
    lo = float(x[0]) if lo_open else cross(a, a - 1)
    hi = float(x[-1]) if hi_open else cross(b, b + 1)
    return dict(lo=lo, hi=hi, lo_open=bool(lo_open), hi_open=bool(hi_open),
                mode=float(x[int(np.nanargmax(prof))]))


def covers(gf: GridFit, K, theta, N) -> dict:
    """Does each quantity's 95% profile region contain its true value?

    Exact test without interpolation: profile log-lik evaluated AT the true value
    (maximised over the other parameters on the grid) vs. ll_max - 1.92."""
    thr = gf.ll_max - CHI2_1_95
    th = gf.ss.thetas
    res = {}
    res["K"] = bool(ll_given_K_theta(gf, np.full(len(th), K)).max() >= thr)
    res["beta"] = bool(ll_given_K_theta(gf, K * theta / th).max() >= thr)
    res["r"] = bool(ll_given_K_theta(gf, 1 + theta * (K - 1) / th).max() >= thr)
    # theta, N: interpolate the gridded profiles at the true value
    pt = profile_theta(gf)
    res["theta"] = bool(np.interp(math.log(theta), np.log(th), pt) >= thr)
    pN = profile_N(gf)
    fin = np.isfinite(pN)
    if N <= gf.Ns[fin][0]:
        res["N"] = False
    else:
        res["N"] = bool(np.interp(math.log(N), np.log(gf.Ns[fin]), pN[fin]) >= thr)
    return res


def profile_at_truth(gf: GridFit, K, theta, N) -> dict:
    """Profile log-likelihood of each quantity evaluated at its true value."""
    th = gf.ss.thetas
    pN = profile_N(gf)
    fin = np.isfinite(pN)
    return dict(
        K=float(ll_given_K_theta(gf, np.full(len(th), K)).max()),
        beta=float(ll_given_K_theta(gf, K * theta / th).max()),
        r=float(ll_given_K_theta(gf, 1 + theta * (K - 1) / th).max()),
        theta=float(np.interp(math.log(theta), np.log(th), profile_theta(gf))),
        N=float(-np.inf if N < gf.Ns[fin][0] else np.interp(math.log(N), np.log(gf.Ns[fin]), pN[fin])),
    )


def box_max(gf: GridFit, K_max: float) -> dict:
    """Maximum of the likelihood subject to K <= K_max (the log-lik is concave in K for
    fixed (theta, N), so the constrained optimum is K = min(K_hat, K_max))."""
    ss = gf.ss
    n = ss.n
    Kc = np.minimum(gf.K_hat_thN, K_max)
    ll = (n * np.log(Kc) - Kc * (ss.C0[:, None] - ss.C1[:, None] / gf.Ns[None, :])
          + (n * np.log(ss.thetas) + ss.S_A)[:, None] + gf.D[None, :])
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return dict(ll=float(ll[i, j]), K=float(Kc[i, j]), theta=float(ss.thetas[i]), N=float(gf.Ns[j]),
                K_at_bound=bool(Kc[i, j] >= K_max * 0.999))


# ---------------------------------------------------------------------------
# Independent simulators
# ---------------------------------------------------------------------------


def simulate_hawkesn_inversion(K: float, theta: float, N: float, I0: int, T: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Exact HawkesN sampler by inverting the compensator (no thinning).

    Between events the intensity is  d*K*theta*A*exp(-theta*s)  with d the current
    depletion factor and A the kernel sum just after the last event, so the
    compensator from the last event is  d*K*A*(1 - exp(-theta*s))  and saturates at
    d*K*A: with probability exp(-d*K*A) no further event ever occurs."""
    events = []
    t = 0.0
    A = float(I0)
    used = I0
    log = math.log
    while True:
        d = 1.0 - used / N
        if d <= 0.0:
            break
        E = rng.exponential()
        cap = d * K * A
        if E >= cap:
            break
        s = -log(1.0 - E / cap) / theta
        t += s
        if t >= T:
            break
        events.append(t)
        A = A * math.exp(-theta * s) + 1.0
        used += 1
    return np.asarray(events, dtype=np.float64)


def simulate_sir_gillespie(beta: float, gamma: float, N: int, I0: int, T: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Markovian stochastic SIR (the upstream tutorial's generator); returns the
    times of NEW INFECTIONS in (0, T]."""
    S, I = N - I0, I0
    t = 0.0
    out = []
    while I > 0:
        ri = beta * S * I / N
        rr = gamma * I
        tot = ri + rr
        t += rng.exponential(1.0 / tot)
        if t >= T:
            break
        if rng.random() < ri / tot:
            S -= 1
            I += 1
            out.append(t)
        else:
            I -= 1
    return np.asarray(out, dtype=np.float64)
