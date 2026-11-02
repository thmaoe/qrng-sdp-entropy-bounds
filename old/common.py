"""Helpers shared by the entropy lower-bound modules (HminLower, shannonLower)."""

import chaospy
import numpy as np


def get_rhos(delta):
    """States sent for x = 0 and x = 1, as 2x2 density matrices with overlap `delta`.

    Parameters
    ----------
    delta : float
        Overlap between the two states sent.

    Returns
    -------
    dict
        {0: rho0, 1: rho1}.
    """
    rho0 = np.array([[1., 0], [0, 0]])
    rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)],
                     [delta*np.sqrt(1-delta**2), 1-delta**2]])
    rho1 = rho1 / np.trace(rho1)  # because of floating-point errors
    return {0: rho0, 1: rho1}


def get_quadrature(m_in=4):
    """Gauss-Radau quadrature on [0, 1] with the fixed node t = 1, used by BFF.

    Parameters
    ----------
    m_in : int
        Order of the quadrature; the BFF parameter is m = 2*m_in.

    Returns
    -------
    m : int
        Number of nodes.
    t : np.ndarray
        Nodes t_i.
    w : np.ndarray
        Weights w_i.
    """
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in, distribution, 1.0)
    t = t[0]
    return m, t, w


def get_cm(m, t, w):
    """Quadrature constant c_m = sum_{i < m} w_i / (t_i ln 2).

    Parameters
    ----------
    m : int
        Number of quadrature nodes.
    t, w : np.ndarray
        Quadrature nodes and weights from `get_quadrature`.

    Returns
    -------
    float
        The constant c_m.
    """
    cm = 0.0
    for i in range(m-1):
        cm += w[i]/(t[i]*np.log(2))
    return cm
