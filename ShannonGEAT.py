"""Lower bounds on the smooth min-entropy using the Generalized Entropy Accumulation Theorem (GEAT)."""

import numpy as np


def _geat_bound(Hs, N, nB):
    """Apply the GEAT second-order corrections to a list of single-round entropy values.

    Parameters
    ----------
    Hs : list of float
        Value of the min-tradeoff function for each probability distribution considered.
    N : int
        Number of rounds.
    nB : int
        Number of outcomes.

    Returns
    -------
    float
        GEAT lower bound on the smooth min-entropy rate.
    """
    eps = 1e-8                   # smoothing parameter
    alpha = 1 + 1/np.sqrt(N)     # Renyi parameter

    Min = np.min(Hs)
    Var = np.var(Hs)
    Max = np.max(Hs)
    g = -np.log(1-np.sqrt(1-eps**2))
    V = np.log(2*nB**2 + 1) + np.sqrt(2 + Var)
    K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)+Max-Min))*np.log(2**(2*np.log(nB)+Max-Min) + np.e**2)**3

    return Min - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K


def getH3(ps, N, lambdas, Rs, cm, nB=2, hsrc=0):
    """GEAT lower bound using the Shannon-entropy dual (shannonLower.getHDual).

    Parameters
    ----------
    ps : list of dict
        Probability distributions p[b][x] over which the min-tradeoff function is evaluated.
    N : int
        Number of rounds.
    lambdas : list of dict
        Lagrange multipliers lambdas[i][b][x] returned by getHDual.
    Rs : list of np.ndarray
        Lagrange multipliers (matrices) returned by getHDual.
    cm : float
        BFF constant returned by getHDual.
    nB : int
        Number of outcomes.
    hsrc : float
        Entropy penalty from the source (classical fluctuations).

    Returns
    -------
    float
        GEAT lower bound.
    """
    Hs = []
    for p in ps:
        h = 0.0
        for i, lambs in enumerate(lambdas):
            for b in range(3):
                for x in range(2):
                    h += lambs[b][x] * p[b][x]
            h += np.trace(Rs[i])
        h = cm - h - hsrc
        Hs.append(h)

    return _geat_bound(Hs, N, nB)


def getH3min(ps, N, mus, w, nB=2, hsrc=0):
    """GEAT lower bound using the H_min dual (HminLower.getHminGEAT).

    Parameters
    ----------
    ps : list of dict
        Probability distributions p[b][x] over which the min-tradeoff function is evaluated.
    N : int
        Number of rounds.
    mus : dict
        Dual variables mus[b][x] returned by getHminGEAT.
    w : float
        Point at which -log2 is linearised (typically the guessing probability).
    nB : int
        Number of outcomes.
    hsrc : float
        Entropy penalty subtracted from the source (classical flutuations).

    Returns
    -------
    float
        GEAT lower bound.
    """
    Hs = []
    for p in ps:
        h = 0.0
        for b in range(3):
            for x in range(2):
                h += mus[b][x] * p[b][x]
        res = (w-h) / (w * np.log(2)) - np.log2(w)  # tangent of -log2 at w
        Hs.append(res-hsrc)

    return _geat_bound(Hs, N, nB)
