"""Final-size distribution via sparse Markov chain over (S, I) states.

For the SIR / HawkesN model, the final size distribution can be computed
exactly via a continuous-time Markov chain (CTMC) over the discrete state
space (S, I).  At each state:
    - Infection rate: β · S · I / N = K·θ · S · I / N
    - Recovery rate:  γ · I = θ · I

Final size = N - S(∞), i.e. total number of infected.

State space: (S, I) with S ∈ {0,..,N}, I ∈ {0,..,N-S}, S+I ≤ N.
Total states: (N+1)(N+2)/2, which for N≈1300 is ~847k — SPARSE matrix needed.

§11 pre-registration:
    Hypothesis: The CTMC final-size distribution should show a bi-modal
        shape for K near 1 (near-threshold), collapsing to unimodal as
        K increases above ~1.5.
    Prediction: For K=0.8 (sub-critical), >90% probability mass on small
        outbreaks (final size < 5% of N). For K=2.0 (super-critical),
        two modes: small outbreak + large outbreak with size >40% of N.
        For K=3.0, predominantly large outbreaks.
    Falsifier: If K=0.8 produces significant large-outbreak probability
        (>10% mass above 20% of N), the CTMC calculation is incorrect.

Feasibility note (§11):
    N=1300: |states| ≈ (1300+1)(1300+2)/2 ≈ 847k states.
    Transition matrix has ≤ 2 outgoing transitions per state (infection or recovery),
    so nnz ≈ 2 × 847k ≈ 1.7M entries. A CSR sparse matrix of this size requires
    ~25 MB for float64, which is feasible. The final-size distribution is computed
    by absorbing Markov chain analysis (solving a linear system), not by full
    matrix exponentiation.

    HOWEVER, for N=1300 the state space is 847k and solving the linear system
    for absorption probabilities may be slow (~minutes). We implement a
    Monte Carlo approximation for large N and use the CTMC for smaller N.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _state_index(S: int, I: int, N: int) -> int:
    """Map (S, I) to a linear index.

    States are ordered by (N-S) then I, i.e. row-by-row where row k
    has states with S = N-k, I = 0..k.
    """
    # Row k = N - S; column = I; index = k*(k+1)//2 + I
    k = N - S
    return k * (k + 1) // 2 + I


def _n_states(N: int) -> int:
    return (N + 1) * (N + 2) // 2


def final_size_distribution_mc(
    K: float,
    theta: float,
    N: int,
    I0: int = 1,
    n_simulations: int = 10_000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate final-size distribution by direct Monte Carlo simulation.

    Simulates the stochastic SIR Gillespie process.

    Parameters
    ----------
    K, theta : HawkesN / SIR parameters
    N        : population size
    I0       : initial infected count
    n_simulations : number of Gillespie runs
    rng_seed : RNG seed

    Returns
    -------
    final_sizes : array of final sizes (R(∞) = N - S(∞)), shape (n_simulations,)
    bins        : bin edges [0, 1, ..., N]
    """
    rng = np.random.default_rng(rng_seed)
    beta = K * theta  # infection rate
    gamma = theta     # recovery rate

    final_sizes = np.zeros(n_simulations, dtype=np.int32)

    for sim_i in range(n_simulations):
        S = N - I0
        I = I0
        R = 0

        while I > 0:
            infection_rate = beta * S * I / N
            recovery_rate = gamma * I
            total_rate = infection_rate + recovery_rate

            if total_rate <= 0:
                break

            # Draw time to next event (exponential)
            dt = rng.exponential(1.0 / total_rate)

            # Determine event type
            if rng.uniform() < infection_rate / total_rate:
                S -= 1
                I += 1
            else:
                I -= 1
                R += 1

        final_sizes[sim_i] = R  # = N - S

    bins = np.arange(N + 2, dtype=np.float64)  # [0, 1, ..., N+1]
    return final_sizes, bins


def final_size_distribution_ctmc(
    K: float,
    theta: float,
    N: int,
    I0: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact final-size distribution via absorbing CTMC.

    For small N (≤ 200), computes exact absorption probabilities.
    For larger N, raises ValueError and users should call the MC version.

    Uses the fact that in the stochastic SIR (= HawkesN mean-field limit),
    the Markov chain on (S, I) is absorbed at states with I = 0.

    The absorption probability at each state gives P(final_size = N - S).

    Parameters
    ----------
    K, theta : parameters
    N        : population size (must be ≤ 200 for this function)
    I0       : initial infected count (default 1)

    Returns
    -------
    pmf   : probability mass at each final size R ∈ {0, 1, ..., N}, shape (N+1,)
    sizes : final size values [0, ..., N]
    """
    if N > 300:
        raise ValueError(f"N={N} too large for CTMC; use final_size_distribution_mc")

    beta = K * theta / N  # per-pair infection rate (so total = beta * S * I)
    gamma_rate = theta    # recovery rate per infected

    n_states = _n_states(N)

    # Build sparse transition rate matrix Q (column = from, row = to)
    # We'll actually build Q as Q[from, to] then transpose
    rows = []
    cols = []
    data = []

    for S in range(N + 1):
        for I in range(N - S + 1):
            if I == 0:
                continue  # absorbing state
            from_idx = _state_index(S, I, N)

            # Infection event: S -> S-1, I -> I+1
            if S > 0:
                inf_rate = beta * S * I
                to_idx = _state_index(S - 1, I + 1, N)
                rows.append(from_idx)
                cols.append(to_idx)
                data.append(inf_rate)

            # Recovery event: I -> I-1
            rec_rate = gamma_rate * I
            to_idx = _state_index(S, I - 1, N)
            rows.append(from_idx)
            cols.append(to_idx)
            data.append(rec_rate)

    # Build rate matrix (rows=from, cols=to, values=rate)
    Q = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_states, n_states),
        dtype=np.float64,
    )

    # For absorbing Markov chains, the absorption probabilities from
    # transient states can be computed from the fundamental matrix.
    # Here we use a simpler backward-pass approach: compute the probability
    # of ending in each absorbing state from the initial state.

    # Identify absorbing states: I = 0, so state = _state_index(S, 0, N) for S in 0..N
    absorbing = {_state_index(S, 0, N): S for S in range(N + 1)}

    # Compute probabilities by solving: for each absorbing state s_a,
    # P(absorbed at s_a | start at s_0) satisfies a linear system.
    # This is O(n_states²) in general.

    # Alternative: compute the full absorption probability vector by
    # simulating/solving the Kolmogorov forward equations to stationarity.
    # For small N, this is feasible.

    # We use the fact that for an absorbing CTMC, the probability vector
    # p(t) → steady state at t→∞, which lives on the absorbing states.
    # We can compute it via the embedded chain or by solving the system:
    #   p_T · Q_T = 0  (for transient states)
    # where Q_T is the sub-rate-matrix on transient states.

    # Let T = transient states (I > 0), A = absorbing states (I = 0)
    # Partition Q: Q_TT (transient→transient), Q_TA (transient→absorbing)
    # Absorption probabilities: B = -Q_TT^{-1} · Q_TA
    # Row i of B = absorption probs from transient state i.

    transient_idx = []
    absorbing_idx = []
    for S in range(N + 1):
        for I in range(N - S + 1):
            idx = _state_index(S, I, N)
            if I == 0:
                absorbing_idx.append((idx, S))
            else:
                transient_idx.append(idx)

    t_size = len(transient_idx)
    a_size = len(absorbing_idx)
    t_map = {global_i: local_i for local_i, global_i in enumerate(transient_idx)}
    a_map = {global_i: local_i for local_i, (global_i, _) in enumerate(
        zip([v[0] for v in absorbing_idx], absorbing_idx)
    )}

    # Build Q_TT and Q_TA
    # Rate out of each transient state (diagonal of -Q_TT)
    # We need to iterate over transitions
    QTT_rows = []
    QTT_cols = []
    QTT_data = []
    QTA_rows = []
    QTA_cols = []
    QTA_data = []

    outflow = np.zeros(t_size, dtype=np.float64)

    for S in range(N + 1):
        for I in range(1, N - S + 1):
            from_global = _state_index(S, I, N)
            from_local = t_map[from_global]

            # Infection: to (S-1, I+1)
            if S > 0:
                inf_rate = beta * S * I
                to_global = _state_index(S - 1, I + 1, N)
                outflow[from_local] += inf_rate
                if I + 1 > 0 and S - 1 >= 0:
                    # Is the destination transient or absorbing?
                    # I+1 > 0 always (since I >= 1), so it's transient
                    to_local = t_map[to_global]
                    QTT_rows.append(from_local)
                    QTT_cols.append(to_local)
                    QTT_data.append(inf_rate)

            # Recovery: to (S, I-1)
            rec_rate = gamma_rate * I
            to_global = _state_index(S, I - 1, N)
            outflow[from_local] += rec_rate
            if I - 1 == 0:
                # Absorbing state: (S, 0)
                to_local_a = None
                for local_a, (glob_a, S_a) in enumerate(absorbing_idx):
                    if glob_a == to_global:
                        to_local_a = local_a
                        break
                if to_local_a is not None:
                    QTA_rows.append(from_local)
                    QTA_cols.append(to_local_a)
                    QTA_data.append(rec_rate)
            else:
                to_local = t_map[to_global]
                QTT_rows.append(from_local)
                QTT_cols.append(to_local)
                QTT_data.append(rec_rate)

    # Build Q_TT as rate matrix (off-diagonal = rate, diagonal = -outflow)
    QTT = sparse.csr_matrix(
        (QTT_data, (QTT_rows, QTT_cols)),
        shape=(t_size, t_size),
        dtype=np.float64,
    )
    # Subtract diagonal (outflow)
    QTT = QTT - sparse.diags(outflow, 0, shape=(t_size, t_size), format="csr")

    QTA = sparse.csr_matrix(
        (QTA_data, (QTA_rows, QTA_cols)),
        shape=(t_size, a_size),
        dtype=np.float64,
    )

    # Solve: -Q_TT · B = Q_TA  => B = (-Q_TT)^{-1} · Q_TA
    # Column j of B = absorption probs into absorbing state j
    # We need one row of B: row corresponding to initial state (N - I0, I0)
    init_global = _state_index(N - I0, I0, N)
    init_local = t_map[init_global]

    # Solve for the init_local row of B by solving:
    # (-Q_TT)^T · b_row = e_{init_local}
    # which gives B[init_local, :]
    # Actually we want B[init_local, :], where B = (-Q_TT)^{-1} · Q_TA
    # Equivalently solve (-Q_TT) · b = QTA[init_local, :]^T... no.
    #
    # The i-th row of B satisfies: Σ_j (-Q_TT)[i,j] B[j,:] = QTA[i,:]
    # So for the i-th row: we want B[i, :] = (-Q_TT)^{-1} [i-th row] * QTA
    # It's easier to compute the whole thing as:
    #   B_row = e_i^T · (-Q_TT)^{-1} · QTA
    # where e_i is a unit vector.

    # We'll solve the sparse system -Q_TT^T · v = e_i, then B_row = v^T · QTA
    e_i = np.zeros(t_size, dtype=np.float64)
    e_i[init_local] = 1.0

    neg_QTT_T = (-QTT).T
    v = spsolve(neg_QTT_T, e_i)
    B_row = v @ QTA  # shape (a_size,)

    # Map absorption probabilities to final sizes
    pmf = np.zeros(N + 1, dtype=np.float64)
    for local_a, (glob_a, S_a) in enumerate(absorbing_idx):
        final_size = N - S_a  # R = N - S (since I=0 at absorption)
        pmf[final_size] += B_row[local_a]

    # Normalise (should already sum to ~1)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf /= pmf_sum

    sizes = np.arange(N + 1)
    return pmf, sizes


def final_size_distribution(
    K: float,
    theta: float,
    N: int,
    I0: int = 1,
    n_mc: int = 20_000,
    rng_seed: int = 0,
    ctmc_max_n: int = 200,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute final-size distribution, choosing CTMC or MC based on N.

    Parameters
    ----------
    K, theta : model parameters
    N        : population size
    I0       : initial infected
    n_mc     : MC simulations (used when N > ctmc_max_n)
    rng_seed : RNG seed for MC
    ctmc_max_n : threshold; use CTMC for N <= this, MC otherwise

    Returns
    -------
    pmf    : probability mass function over final sizes, shape (N+1,)
    sizes  : final size values [0..N]
    method : 'ctmc' or 'mc'
    """
    if N <= ctmc_max_n:
        pmf, sizes = final_size_distribution_ctmc(K, theta, N, I0)
        return pmf, sizes, "ctmc"
    else:
        final_sizes, bins = final_size_distribution_mc(
            K, theta, N, I0, n_mc, rng_seed
        )
        pmf, _ = np.histogram(final_sizes, bins=bins, density=False)
        pmf = pmf.astype(np.float64) / pmf.sum()
        sizes = np.arange(N + 1)
        return pmf, sizes, "mc"
