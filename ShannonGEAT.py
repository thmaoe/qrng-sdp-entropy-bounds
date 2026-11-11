"""Lower bounds on the smooth min-entropy using the Generalized Entropy Accumulation Theorem (GEAT)."""

import numpy as np
from shannonLower import getHDual
from HminLower import getHminGEAT



def _geat_bound(fs, N, nB, f_ref, pct):
    """Apply the GEAT second-order corrections to a list of single-round entropy values.

    Parameters
    ----------
    fs : list of float
        Value of the min-tradeoff function for each frequencies considered.
    N : int
        Number of rounds.
    nB : int
        Number of outcomes.
    f_ref: float
        Reference fmin to compute h_th.
    pct: float in (0,1)
        pourcentage of f_ref to use for h_th.

    Returns
    -------
    float
        GEAT lower bound on the smooth min-entropy rate.
    """
    eps = 1e-8                   # smoothing parameter
    alpha = 1 + 1/np.sqrt(N)     # Renyi parameter
    eps_acc = 1e-8

    Min = np.min(fs)
    Var = np.var(fs)
    Max = np.max(fs)
    g = -np.log2(1-np.sqrt(1-eps**2))
    V = np.log2(2*nB**2 + 1) + np.sqrt(2 + Var)
    K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log2(nB)+Max-Min))*np.log(2**(2*np.log2(nB)+Max-Min) + np.e**2)**3

    h_th = pct * f_ref

    return h_th - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - (g + alpha*np.log2(1/eps_acc))/(N*(alpha-1)) - ((alpha-1)/(2-alpha))**2 * K

def getfBFF(p, lambdas, Rs, cm, hsrc=0):
    """Min tradeoff function f for BFF.

    Parameters
    ----------
    p : dict
        Probability distributions p[b][x] over which the min-tradeoff function is evaluated.
    lambdas : list of dict
        Lagrange multipliers lambdas[i][b][x] returned by getHDual.
    Rs : list of np.ndarray
        Lagrange multipliers (matrices) returned by getHDual.
    cm : float
        BFF constant returned by getHDual.
    hsrc : float
        Entropy penalty from the source (classical fluctuations).

    Returns
    -------
    float
        Min tradeoff function value for BFF route.
    """
    f = 0.0
    for i, lambs in enumerate(lambdas):
        for b in range(3):
            for x in range(2):
                f += lambs[b][x] * p[b][x]
        f += np.trace(Rs[i])
    f = cm - f - hsrc
    return f


def getH3(ps, N, lambdas, Rs, cm, f_ref, pct=0.95, hsrc=0, nB=2):
    """GEAT lower bound using the BFF route.

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
    fs = [] #min tradeoff function for each frequencies in ps.
    for p in ps:
        fs.append(getfBFF(p, lambdas, Rs, cm, hsrc))
    return _geat_bound(fs, N, nB, f_ref, pct)

def getfHmin(p, mus, w, hsrc=0):
    """Min tradeoff function for Hmin route.

    Parameters
    ----------
    p : dict
        Probability distributions p[b][x] over which the min-tradeoff function is evaluated.
    mus : dict
        Dual variables mus[b][x] returned by getHminGEAT.
    w : float
        Point at which -log2 is linearised (typically the guessing probability).
    hsrc : float
        Entropy penalty subtracted from the source (classical flutuations).

    Returns
    -------
    float
        Min tradeoff function for Hmin route.
    """
    h = 0.0
    for b in range(3):
        for x in range(2):
            h += mus[b][x] * p[b][x]
    h = (w-h) / (w * np.log(2)) - np.log2(w)  # tangent of -log2 at w

    return (h - hsrc)[0]

def getH3min(ps, N, mus, w, f_ref, pct=0.95, hsrc=0, nB=2):
    """GEAT lower bound using the H_min route.

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
    fs = []
    for p in ps:
        fs.append(getfHmin(p, mus, w, hsrc))

    return _geat_bound(fs, N, nB, f_ref, pct)


def Hgeat(ps, delta, nPoints, pct, route, px, hsrc=0, verbose=False):
    """Full wrapped function to get GEAT lower bound.
    
        Parameters
        ----------
        ps : list of dict
            Probability distributions p[b][x] over which the min-tradeoff function is evaluated. The first one is used for reference.
        delta : float
            Overlap between states for given experiment.
        nPoints : int
            Number of points in each blocks.
        pct : float in (0,1)
            Percentage of min tradeoff function to use.
        route : string
            Wether to use "bff" or "hmin" route. 
        px : float in (0,1)
            Probability that x=1.
        hsrc: float
            Entropy penalty from the source (classical fluctuations).
        verbose: bool
            Wether to print how many blocks and which were used. 
            
    
        Returns
        -------
        float
            GEAT lower bound.
        """
    
    if len(ps) <= 1:
        raise RuntimeError("need more than one block of data!")
    
    if route == "bff":
        _, lambdas, Rs, cm = getHDual(delta, ps[0], px)
        f_ref = getfBFF(ps[0], lambdas, Rs, cm, hsrc)
        ps_use = []
        ids = []
        for i, p in enumerate(ps[1:]):
            if (pct * f_ref <= getfBFF(p, lambdas, Rs, cm, hsrc)):
                ps_use.append(p)
                ids.append(i)
        if len(ps_use) == 0:
            raise RuntimeError("Blocks not good enough! Redo experiment or change pct!")
        if verbose: print(f"Using {len(ps_use)} blocks out of {len(ps)-1} (blocks {ids})")
        return getH3(ps_use, nPoints, lambdas, Rs, cm, f_ref, pct, hsrc)
    
    elif route == "hmin":
        w, mus = getHminGEAT(ps[0], delta, px)
        f_ref = getfHmin(ps[0], mus, w, hsrc)
        ps_use = []
        ids = []
        for i, p in enumerate(ps[1:]):
            if (pct * f_ref <= getfHmin(p, mus, w, hsrc)):
                ps_use.append(p)
                ids.append(i)
        if len(ps_use) == 0:
            raise RuntimeError("Blocks not good enough! Redo experiment or change pct!")
        
        if verbose: print(f"Using {len(ps_use)} blocks out of {len(ps)-1} (blocks {ids})")
        return getH3min(ps_use, nPoints, mus, w, f_ref, pct, hsrc)

    else: 
        raise RuntimeError("Wrong route identifier: try hmin or bff")


