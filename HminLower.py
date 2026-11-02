"""Lower bound on the min-entropy H_min.

By default Eve guesses the coarse-grained variable c (inconclusive / conclusive).
With guess_b=True she guesses the 3 outcomes b instead (this does not change anything for Photon Number ('0') and BPSK ('2')).

Only for the scenario without an input y for Bob (impl = 0 in shannonLower).
"""

import itertools

import cvxpy as cp
import numpy as np

from common import get_rhos

# Map from outcome b to the value Eve guesses, and order in which the constraint blocks are built
GUESS_C = {2: 0, 0: 1, 1: 1}  # c: b = 2 (inconclusive) -> 0, b in {0, 1} (conclusive) -> 1
ORDER_C = (2, 0, 1)
GUESS_B = {0: 0, 1: 1, 2: 2}  # b itself
ORDER_B = (0, 1, 2)


def _solve_hmin_sdp(p, delta, px, guess_b=False):
    """Build and solve the H_min SDP shared by `getHmin` and `getHminGEAT`.

    For every outcome b and every guess pair g = (g0, g1) (Eve guesses g_x when the input is x):

        sum_x rho_x * (px_x * [g_x == guess(b)] - mu_bx) + H_g - 1/2 tr(H_g) id  <=  0

    with px_0 = 1 - px, px_1 = px, and guess(b) given by GUESS_C or GUESS_B.

    Returns the solved cvxpy problem and the dual variables mu[b][x] (as cvxpy Variables).
    """
    guess, order = (GUESS_B, ORDER_B) if guess_b else (GUESS_C, ORDER_C)
    n_guess = len(set(guess.values()))

    H = {g: cp.Variable((2, 2), hermitian=True) for g in itertools.product(range(n_guess), repeat=2)}
    mu = {b: {x: cp.Variable(1) for x in (0, 1)} for b in (2, 0, 1)}

    rho = get_rhos(delta)
    pxs = [1-px, px]

    constraints = []
    for b in order:
        for g, Hg in H.items():
            terms = [rho[x] * (pxs[x] - mu[b][x]) if g[x] == guess[b] else -rho[x] * (mu[b][x])
                     for x in (0, 1)]
            constraints += [(terms[0] + terms[1] + Hg - 0.5*cp.trace(Hg)*np.eye(2)) << 0]

    obj = cp.real(mu[2][0] * p[2][0] + mu[2][1] * p[2][1] + mu[1][0] * p[1][0] + mu[1][1] * p[1][1] + mu[0][1] * p[0][1] + mu[0][0] * p[0][0])

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver="MOSEK")

    return prob, mu


def getHmin(p, delta, px, guess_b=False):
    """Lower bound on H_min (no input y for Bob; any implementation '0' to '3').

    Parameters
    ----------
    p : dict
        Probabilities p[b][x], b in {0, 1, 2} (2 = inconclusive), x in {0, 1}.
    delta : float
        Overlap between the two states sent.
    px : float
        Probability of input x = 1.
    guess_b : bool
        False (default): Eve guesses c (inconclusive / conclusive).
        True: Eve guesses the 3 outcomes b.

    Returns
    -------
    float
        -log2 of the optimal guessing probability.
    """
    prob, _ = _solve_hmin_sdp(p, delta, px, guess_b)
    return -np.log2(prob.value)


def getHminGEAT(p, delta, px, guess_b=False):
    """Same SDP as `getHmin`, returning what is needed for the GEAT.

    Parameters
    ----------
    p, delta, px, guess_b :
        See `getHmin`.

    Returns
    -------
    value : float
        Optimal guessing probability (not its -log2).
    mus : dict
        Optimal dual variables mus[b][x].
    """
    prob, mu = _solve_hmin_sdp(p, delta, px, guess_b)
    return prob.value, {b: {x: mu[b][x].value for x in (0, 1)} for b in (2, 1, 0)}
