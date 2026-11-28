"""Calibration machinery for the WP-02 completion study (2026-11).

For one "cell" (fixed true SIR parameters, bin layout, noise generator) this
module simulates many replicate datasets and, for each, computes

* a maximum-likelihood fit that is NOT initialised at the truth, with
  Laplace (inverse observed Hessian) intervals pushed to derived targets by the
  delta method, and
* a full posterior under flat priors on the fit coordinates, by self-normalised
  importance sampling with the scale parameter integrated out analytically.

Assumptions that are surfaced on purpose (method-discipline §7):

* Observation model for fitting: y_j ~ Poisson(s * int_bin I dt), independent
  bins, deterministic SIR mean. Real post counts are almost certainly
  overdispersed; ``run_e3_noise`` tests what that does.
* Prior for the posterior: uniform on (log beta, log gamma, logit I0) inside
  the box the first pass used (``BOX3``), and p(log s) flat and unbounded. With
  N >= 1 observed posts the posterior is proper and s | rest ~ Gamma(N, M)
  where M = sum_j int_bin I dt. The first pass bounded log s to
  [log 1e-3, log 1e8]; that truncation is ignored here.
* The importance-sampling proposal is centred on the TRUE parameters of the
  cell (plus a prior-wide component). This uses the truth only to place
  quadrature nodes; the target posterior does not depend on it. The effective
  sample size per dataset is recorded so that numerical failure is visible.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln as np_gammaln
from scipy.special import lambertw, ndtr

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from experiments.wp02.sir import bin_means

jax.config.update("jax_enable_x64", True)

# (log beta, log gamma, logit I0) prior box; identical to the first pass.
BOX3 = np.array(
    [
        [np.log(0.03), np.log(30.0)],
        [np.log(0.01), np.log(10.0)],
        [-14.0, 0.0],
    ]
)
BOX4 = np.array(
    [
        [np.log(0.03), np.log(30.0)],
        [np.log(0.01), np.log(10.0)],
        [np.log(1e-3), np.log(1e8)],
        [-14.0, 0.0],
    ]
)

TARGETS = (
    "log_beta",
    "log_gamma",
    "logit_i0",
    "log_scale",
    "log_R0",
    "growth_rate",
    "peak_time",
    "log_peak_rate",
    "log_total",
    "log_init_rate",
)
TARGET_LABELS = {
    "log_beta": "log β (infection rate)",
    "log_gamma": "log γ (recovery rate)",
    "logit_i0": "logit I0 (initial fraction)",
    "log_scale": "log s (scale)",
    "log_R0": "log R0 (= β/γ)",
    "growth_rate": "initial growth rate β(1−I0)−γ  [1/time unit]",
    "peak_time": "peak time [time units]",
    "log_peak_rate": "log peak height (posts per time unit)",
    "log_total": "log final size (total posts, whole curve)",
    "log_init_rate": "log initial post rate s·I0",
}

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


# --------------------------------------------------------------------------
# Derived curve summaries (numpy, vectorised)
# --------------------------------------------------------------------------
def derived(log_beta, log_gamma, logit_i0, log_scale):
    """Return array (..., len(TARGETS)) of derived targets.

    Peak time/height are of the continuous curve s*I(t). Subcritical curves
    (beta*S0 <= gamma) peak at t=0 with height s*I0.
    """
    log_beta, log_gamma, logit_i0, log_scale = np.broadcast_arrays(
        np.asarray(log_beta, float),
        np.asarray(log_gamma, float),
        np.asarray(logit_i0, float),
        np.asarray(log_scale, float),
    )
    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    i0 = 1.0 / (1.0 + np.exp(-logit_i0))
    s0 = 1.0 - i0
    r0 = beta / gamma
    growth = beta * s0 - gamma
    supercrit = growth > 0

    # peak height: I_max = 1 - S* + (gamma/beta) ln(S*/S0), S* = gamma/beta
    safe_r0 = np.where(supercrit, r0, 2.0)
    safe_s0 = np.where(supercrit, s0, 0.9)
    sstar = 1.0 / safe_r0
    imax = 1.0 - sstar + sstar * np.log(sstar / safe_s0)
    imax = np.where(supercrit, imax, i0)

    # peak time by Gauss-Legendre after x = ln S0 - delta (e^u - 1)
    safe_i0 = np.where(supercrit, i0, 0.1)
    x0 = np.log(safe_s0)
    x1 = np.log(sstar)
    delta = safe_i0 / (safe_s0 - sstar)
    big_u = np.log1p((x0 - x1) / delta)
    u = 0.5 * big_u[..., None] * (_GL_X + 1.0)
    dx = delta[..., None] * np.expm1(u)
    x = x0[..., None] - dx
    inf = -np.expm1(x) - sstar[..., None] * dx  # I(x) = 1 - e^x + (g/b)(x - x0)
    inf = np.maximum(inf, 1e-300)
    integrand = delta[..., None] * np.exp(u) / inf
    safe_beta = np.where(supercrit, beta, 1.0)
    tpeak = 0.5 * big_u * np.sum(integrand * _GL_W, axis=-1) / safe_beta
    tpeak = np.where(supercrit, tpeak, 0.0)

    # final size: S_inf = -W0(-R0 S0 e^{-R0}) / R0 ;  int_0^inf I dt = (1 - S_inf)/gamma
    with np.errstate(under="ignore", over="ignore"):
        arg = -r0 * s0 * np.exp(-r0)
    sinf = -np.real(lambertw(arg, 0)) / r0
    area = (1.0 - sinf) / gamma

    out = np.stack(
        [
            log_beta,
            log_gamma,
            logit_i0,
            log_scale,
            log_beta - log_gamma,
            growth,
            tpeak,
            log_scale + np.log(imax),
            log_scale + np.log(area),
            log_scale + np.log(i0),
        ],
        axis=-1,
    )
    return out


def scale_for_total(beta, gamma, i0, total_posts):
    """Scale s such that the expected number of posts over the whole curve is total_posts."""
    d = derived(np.log(beta), np.log(gamma), np.log(i0 / (1 - i0)), 0.0)
    return float(total_posts / np.exp(d[TARGETS.index("log_total")]))


# --------------------------------------------------------------------------
# JAX pieces
# --------------------------------------------------------------------------
def _theta4(theta3, log_scale):
    return jnp.array([theta3[0], theta3[1], log_scale, theta3[2]])


def make_forward(bin_edges):
    """Batched unit-scale bin integrals m_j(theta3), chunked to bound memory."""
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    f = jax.jit(jax.vmap(lambda t3: bin_means(_theta4(t3, 0.0), edges)))

    def forward(theta3, chunk=20000):
        theta3 = np.asarray(theta3, float)
        outs = []
        for start in range(0, len(theta3), chunk):
            block = theta3[start : start + chunk]
            pad = chunk - len(block)
            if pad:
                block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
            outs.append(np.asarray(f(jnp.asarray(block)))[: chunk - pad])
        return np.concatenate(outs)

    return forward


def nll(theta4, counts, edges, invk):
    """Negative log-likelihood. invk = 0: Poisson; invk > 0: negative binomial, var = mu + invk mu^2."""
    mu = bin_means(theta4, edges)
    pois = mu - counts * jnp.log(mu) + gammaln(counts + 1.0)
    k = 1.0 / jnp.where(invk > 0, invk, 1.0)
    nb = -(
        gammaln(counts + k)
        - gammaln(k)
        - gammaln(counts + 1.0)
        + k * jnp.log(k / (k + mu))
        + counts * jnp.log(mu / (k + mu))
    )
    return jnp.sum(jnp.where(invk > 0, nb, pois))


def make_fitter(bin_edges, n_iter=40):
    """Vmapped damped Fisher-scoring fit in the 4 fit coordinates, box-constrained.

    Returns (theta_hat, nll, observed Hessian, expected Fisher, Newton decrement).
    """
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    lo = jnp.asarray(BOX4[:, 0])
    hi = jnp.asarray(BOX4[:, 1])

    def mean_and_jac(theta4):
        jac, mu = jax.jacfwd(lambda t: (bin_means(t, edges),) * 2, has_aux=True)(theta4)
        return mu, jac

    def fit_one(theta0, counts, invk):
        def body(state, _):
            theta, lam, cur = state
            mu, jac = mean_and_jac(theta)
            var = mu + invk * mu**2
            grad = jac.T @ ((mu - counts) / var)
            fisher = jac.T @ (jac / var[:, None])
            damped = fisher + lam * jnp.diag(jnp.diag(fisher)) + 1e-10 * jnp.eye(4)
            step = -jnp.linalg.solve(damped, grad)
            step = jnp.where(jnp.isfinite(step), step, 0.0)
            cand = jnp.clip(theta + step, lo, hi)
            cand_nll = nll(cand, counts, edges, invk)
            ok = jnp.isfinite(cand_nll) & (cand_nll < cur)
            theta = jnp.where(ok, cand, theta)
            cur = jnp.where(ok, cand_nll, cur)
            lam = jnp.clip(jnp.where(ok, lam / 3.0, lam * 4.0), 1e-9, 1e9)
            return (theta, lam, cur), None

        init = (theta0, jnp.asarray(1e-2), nll(theta0, counts, edges, invk))
        (theta, _, cur), _ = jax.lax.scan(body, init, None, length=n_iter)
        mu, jac = mean_and_jac(theta)
        var = mu + invk * mu**2
        grad = jac.T @ ((mu - counts) / var)
        fisher = jac.T @ (jac / var[:, None])
        free = (theta > lo + 1e-4) & (theta < hi - 1e-4)
        g = jnp.where(free, grad, 0.0)
        dec = g @ jnp.linalg.solve(fisher + 1e-8 * jnp.diag(jnp.diag(fisher)) + 1e-12 * jnp.eye(4), g)
        hess = jax.hessian(nll)(theta, counts, edges, invk)
        return theta, cur, hess, fisher, dec, mu

    return jax.jit(jax.vmap(fit_one))


def expected_fisher(theta4_true, bin_edges):
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    t = jnp.asarray(theta4_true)
    jac = np.asarray(jax.jacfwd(lambda x: bin_means(x, edges))(t))
    mu = np.asarray(bin_means(t, edges))
    return jac.T @ (jac / mu[:, None]), mu


# --------------------------------------------------------------------------
# Importance cloud
# --------------------------------------------------------------------------
def build_cloud(rng, centre3, cov3, n_points, inflations=(1.5, 4.0, 12.0), prior_frac=0.25):
    """Mixture proposal: uniform prior box + box-truncated Gaussians about centre3.

    Returns points (K,3) and log(prior/q) up to a constant.
    """
    lo, hi = BOX3[:, 0], BOX3[:, 1]
    vol = np.prod(hi - lo)
    n_comp = len(inflations)
    n_prior = int(round(prior_frac * n_points))
    per = (n_points - n_prior) // n_comp
    chol = np.linalg.cholesky(cov3 + 1e-12 * np.eye(3))
    parts = [rng.uniform(lo, hi, size=(n_prior, 3))]
    accept = []
    for infl in inflations:
        z = rng.standard_normal((200_000, 3))
        inside = np.all(((centre3 + infl * z @ chol.T) > lo) & ((centre3 + infl * z @ chol.T) < hi), axis=1)
        accept.append(max(inside.mean(), 1e-6))
        got = np.empty((0, 3))
        while len(got) < per:
            z = rng.standard_normal((int(per / accept[-1] * 1.3) + 100, 3))
            cand = centre3 + infl * z @ chol.T
            keep = np.all((cand > lo) & (cand < hi), axis=1)
            got = np.concatenate([got, cand[keep]])
        parts.append(got[:per])
    pts = np.concatenate(parts)
    k_total = len(pts)
    dens = np.full(k_total, (n_prior / k_total) / vol)
    cov_inv = np.linalg.inv(cov3 + 1e-12 * np.eye(3))
    logdet = np.linalg.slogdet(cov3 + 1e-12 * np.eye(3))[1]
    diff = pts - centre3
    maha = np.einsum("ki,ij,kj->k", diff, cov_inv, diff)
    for infl, acc in zip(inflations, accept):
        logn = -0.5 * maha / infl**2 - 0.5 * logdet - 3 * np.log(infl) - 1.5 * np.log(2 * np.pi)
        dens += (per / k_total) * np.exp(logn) / acc
    return pts, -np.log(dens)  # prior is constant on the box


def build_cloud_v2(rng, centre3, cov3, n_points, inflations=(1.5, 4.0, 12.0), prior_frac=0.2):
    """As build_cloud, plus Gaussian components in psi = (growth rate, log gamma, logit I0).

    Truncated-window posteriors are slabs of near-constant growth rate, which are curved in
    (log beta, log gamma, logit I0) but flat in psi. Added after E1 showed low effective
    sample sizes for high-count pre-peak cells with the v1 proposal.
    """
    lo, hi = BOX3[:, 0], BOX3[:, 1]
    vol = np.prod(hi - lo)

    def to_psi(t):
        beta, gam = np.exp(t[..., 0]), np.exp(t[..., 1])
        i0 = 1 / (1 + np.exp(-t[..., 2]))
        return np.stack([beta * (1 - i0) - gam, t[..., 1], t[..., 2]], axis=-1), beta * (1 - i0)

    def from_psi(p):
        gam = np.exp(p[..., 1])
        i0 = 1 / (1 + np.exp(-p[..., 2]))
        with np.errstate(invalid="ignore", divide="ignore"):
            logb = np.log((p[..., 0] + gam) / (1 - i0))
        return np.stack([logb, p[..., 1], p[..., 2]], axis=-1)

    b, g = np.exp(centre3[0]), np.exp(centre3[1])
    i0c = 1 / (1 + np.exp(-centre3[2]))
    jac = np.array([[b * (1 - i0c), -g, -b * i0c * (1 - i0c)], [0, 1, 0], [0, 0, 1.0]])
    cov_psi = jac @ cov3 @ jac.T
    centre_psi = to_psi(centre3)[0]
    comps = [("theta", centre3, cov3, f) for f in inflations] + [("psi", centre_psi, cov_psi, f) for f in inflations]
    n_prior = int(round(prior_frac * n_points))
    per = (n_points - n_prior) // len(comps)

    def draw(kind, centre, chol, infl, n):
        x = centre + infl * rng.standard_normal((n, 3)) @ chol.T
        t = x if kind == "theta" else from_psi(x)
        keep = np.all(np.isfinite(t), axis=1)
        keep[keep] &= np.all((t[keep] > lo) & (t[keep] < hi), axis=1)
        return t[keep], keep.mean()

    parts = [rng.uniform(lo, hi, size=(n_prior, 3))]
    info = []
    for kind, centre, cov, infl in comps:
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(3))
        acc = max(draw(kind, centre, chol, infl, 200_000)[1], 1e-6)
        got = np.empty((0, 3))
        while len(got) < per:
            got = np.concatenate([got, draw(kind, centre, chol, infl, int(per / acc * 1.3) + 100)[0]])
        parts.append(got[:per])
        info.append((kind, centre, np.linalg.inv(cov + 1e-12 * np.eye(3)), np.linalg.slogdet(cov + 1e-12 * np.eye(3))[1], infl, acc))
    pts = np.concatenate(parts)
    k_total = len(pts)
    dens = np.full(k_total, (n_prior / k_total) / vol)
    psi, jac_det = to_psi(pts)
    for kind, centre, cov_inv, logdet, infl, acc in info:
        diff = (pts if kind == "theta" else psi) - centre
        maha = np.einsum("ki,ij,kj->k", diff, cov_inv, diff)
        logn = -0.5 * maha / infl**2 - 0.5 * logdet - 3 * np.log(infl) - 1.5 * np.log(2 * np.pi)
        dens += (per / k_total) * np.exp(logn) / acc * (1.0 if kind == "theta" else jac_det)
    return pts, -np.log(dens)


# --------------------------------------------------------------------------
# Data generators
# --------------------------------------------------------------------------
def simulate(rng, mu, n_rep, generator="poisson", nb_k=None, spike_prob=0.0, spike_size=0.0):
    """Replicate count datasets (n_rep, J).

    generator: 'poisson' | 'nb' (gamma-Poisson, var = mu + mu^2/nb_k) | 'spikes'
    (Poisson plus, independently per bin with prob spike_prob, an extra
    Poisson(spike_size) burst).
    """
    mu = np.asarray(mu)
    if generator == "poisson":
        return rng.poisson(mu, size=(n_rep, len(mu))).astype(float)
    if generator == "nb":
        lam = rng.gamma(nb_k, mu / nb_k, size=(n_rep, len(mu)))
        return rng.poisson(lam).astype(float)
    if generator == "spikes":
        y = rng.poisson(mu, size=(n_rep, len(mu)))
        hit = rng.random((n_rep, len(mu))) < spike_prob
        return (y + hit * rng.poisson(spike_size, size=y.shape)).astype(float)
    raise ValueError(generator)


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------
def _pit(draws, truth):
    """Posterior CDF at truth with mid-rank ties; draws (D, T), truth (T,)."""
    return (draws < truth).mean(axis=0) + 0.5 * (draws == truth).mean(axis=0)


def run_cell(
    *,
    beta,
    gamma,
    i0,
    total_posts,
    bin_edges,
    n_rep,
    seed,
    n_cloud=60_000,
    n_draws=4000,
    generator="poisson",
    gen_kwargs=None,
    fit_nb=False,
    forward=None,
    fitter=None,
    proposal="v1",
):
    """Simulate n_rep datasets from one truth and compute MLE/Laplace + posterior summaries.

    Returns a dict of per-replicate arrays (first axis n_rep; target axis in TARGETS order).
    """
    rng = np.random.default_rng(seed)
    bin_edges = np.asarray(bin_edges, float)
    scale = scale_for_total(beta, gamma, i0, total_posts)
    theta4_true = np.array([np.log(beta), np.log(gamma), np.log(scale), np.log(i0 / (1 - i0))])
    theta3_true = theta4_true[[0, 1, 3]]
    truth = derived(theta4_true[0], theta4_true[1], theta4_true[3], theta4_true[2])
    forward = forward or make_forward(bin_edges)
    fitter = fitter or make_fitter(bin_edges)

    fisher_true, mu_true = expected_fisher(theta4_true, bin_edges)
    w, v = np.linalg.eigh(fisher_true)
    cov4 = (v / np.maximum(w, 1e-4)) @ v.T  # sd capped at 100 per eigendirection
    cov3 = cov4[np.ix_([0, 1, 3], [0, 1, 3])]

    y = simulate(rng, mu_true, n_rep, generator, **(gen_kwargs or {}))
    n_obs = y.sum(axis=1)

    # ---- posterior by importance sampling, scale marginalised --------------
    pts, log_pq = (build_cloud if proposal == "v1" else build_cloud_v2)(rng, theta3_true, cov3, n_cloud)
    m = forward(pts)  # (K, J), unit scale
    logm = np.log(m)
    big_m = m.sum(axis=1)
    log_big_m = np.log(big_m)
    d_cloud = derived(pts[:, 0], pts[:, 1], pts[:, 2], 0.0)  # scale-free parts; add log s later
    scale_cols = np.array([t in ("log_scale", "log_peak_rate", "log_total", "log_init_rate") for t in TARGETS])

    n_t = len(TARGETS)
    out = {
        "n_obs": n_obs,
        "ess": np.full(n_rep, np.nan),
        "post_pit": np.full((n_rep, n_t), np.nan),
        "post_q05": np.full((n_rep, n_t), np.nan),
        "post_q25": np.full((n_rep, n_t), np.nan),
        "post_med": np.full((n_rep, n_t), np.nan),
        "post_q75": np.full((n_rep, n_t), np.nan),
        "post_q95": np.full((n_rep, n_t), np.nan),
        "post_p_peak_in_window": np.full(n_rep, np.nan),
        "post_p_supercritical": np.full(n_rep, np.nan),
        "init_theta": np.tile(theta4_true, (n_rep, 1)),
    }
    t_end = bin_edges[-1]
    i_peak = TARGETS.index("peak_time")
    i_growth = TARGETS.index("growth_rate")
    for start in range(0, n_rep, 100):
        sl = slice(start, min(start + 100, n_rep))
        loglik = y[sl] @ logm.T - n_obs[sl, None] * log_big_m[None, :]  # (r, K)
        logw = loglik + log_pq[None, :]
        logw -= logw.max(axis=1, keepdims=True)
        wts = np.exp(logw)
        wts /= wts.sum(axis=1, keepdims=True)
        out["ess"][sl] = 1.0 / (wts**2).sum(axis=1)
        best = np.argmax(loglik, axis=1)
        for local, r in enumerate(range(sl.start, sl.stop)):
            if n_obs[r] < 1:
                continue
            idx = rng.choice(len(pts), size=n_draws, p=wts[local])
            log_s = np.log(rng.gamma(n_obs[r], 1.0, size=n_draws)) - log_big_m[idx]
            draws = d_cloud[idx] + scale_cols[None, :] * log_s[:, None]
            out["post_pit"][r] = _pit(draws, truth)
            q = np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            for name, row in zip(("post_q05", "post_q25", "post_med", "post_q75", "post_q95"), q):
                out[name][r] = row
            out["post_p_peak_in_window"][r] = np.mean(draws[:, i_peak] < t_end)
            out["post_p_supercritical"][r] = np.mean(draws[:, i_growth] > 0)
            b = best[local]
            out["init_theta"][r] = [pts[b, 0], pts[b, 1], np.log(n_obs[r]) - log_big_m[b], pts[b, 2]]

    # ---- MLE + Laplace ------------------------------------------------------
    init = np.clip(out["init_theta"], BOX4[:, 0] + 1e-6, BOX4[:, 1] - 1e-6)
    fits = {}
    variants = [("pois", np.zeros(n_rep))]
    if fit_nb:
        variants.append(("nb", None))
    for tag, invk in variants:
        if invk is None:  # NB: alternate dispersion grid search and mean-parameter scoring
            invk = np.full(n_rep, 0.1)
            theta = init
            grid = np.concatenate([[0.0], np.logspace(-4, 1, 41)])
            for _ in range(3):
                theta, cur, hess, fisher, dec, mu = (np.asarray(a) for a in fitter(jnp.asarray(theta), jnp.asarray(y), jnp.asarray(invk)))
                ll = np.stack([_nb_loglik_np(y, mu, g) for g in grid], axis=1)
                invk = grid[np.argmax(ll, axis=1)]
            theta, cur, hess, fisher, dec, mu = (np.asarray(a) for a in fitter(jnp.asarray(theta), jnp.asarray(y), jnp.asarray(invk)))
        else:
            theta, cur, hess, fisher, dec, mu = (np.asarray(a) for a in fitter(jnp.asarray(init), jnp.asarray(y), jnp.asarray(invk)))
        fits[tag] = _laplace_summary(theta, hess, fisher, dec, mu, y, truth, n_obs, t_end)
        fits[tag]["invk"] = invk
        fits[tag]["nll"] = cur
    for tag, d in fits.items():
        for key, val in d.items():
            out[f"{tag}_{key}"] = val
    out["truth"] = truth
    out["theta4_true"] = theta4_true
    out["mu_true"] = mu_true
    out["y_first"] = y[:8]
    return out


def _nb_loglik_np(y, mu, invk):
    if invk == 0:
        return np.sum(y * np.log(mu) - mu - np_gammaln(y + 1), axis=1)
    k = 1.0 / invk
    return np.sum(
        np_gammaln(y + k) - np_gammaln(k) - np_gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)),
        axis=1,
    )


def _laplace_summary(theta, hess, fisher, dec, mu, y, truth, n_obs, t_end):
    n_rep = len(theta)
    n_t = len(TARGETS)
    est = derived(theta[:, 0], theta[:, 1], theta[:, 3], theta[:, 2])
    # finite-difference Jacobian of targets wrt the 4 fit coordinates
    jac = np.zeros((n_rep, n_t, 4))
    h = 1e-5
    for c in range(4):
        tp, tm = theta.copy(), theta.copy()
        tp[:, c] += h
        tm[:, c] -= h
        jac[:, :, c] = (derived(tp[:, 0], tp[:, 1], tp[:, 3], tp[:, 2]) - derived(tm[:, 0], tm[:, 1], tm[:, 3], tm[:, 2])) / (2 * h)
    res = {"est": est}
    hess = 0.5 * (hess + np.swapaxes(hess, 1, 2))
    ok_data = n_obs >= 1
    for tag, mat in (("obs", hess), ("exp", fisher)):
        mat = np.where(np.isfinite(mat), mat, 0.0)
        eig = np.linalg.eigvalsh(mat)
        pd = (eig[:, 0] > 0) & ok_data
        cov = np.linalg.pinv(mat + 1e-8 * np.eye(4), hermitian=True)
        var = np.einsum("rtc,rcd,rtd->rt", jac, cov, jac)
        se = np.sqrt(np.clip(var, 0, None))
        se[~pd] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (truth[None, :] - est) / se
        res[f"se_{tag}"] = se
        res[f"pit_{tag}"] = ndtr(z)
        res[f"pd_{tag}"] = pd
        res[f"cond_{tag}"] = np.where(eig[:, 0] > 0, eig[:, -1] / np.where(eig[:, 0] > 0, eig[:, 0], 1.0), np.inf)
    res["boundary"] = np.any((theta < BOX4[:, 0] + 1e-4) | (theta > BOX4[:, 1] - 1e-4), axis=1)
    res["newton_dec"] = dec
    res["theta"] = theta
    dof = max(y.shape[1] - 4, 1)
    res["pearson_phi"] = np.sum((y - mu) ** 2 / mu, axis=1) / dof
    res["peak_in_window"] = est[:, TARGETS.index("peak_time")] < t_end
    return res
