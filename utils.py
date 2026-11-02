"""Miscellaneous helper functions."""

import qutip as qt


def handle_non_phys(p, alpha, impl='1'):
    """Fix probabilities that are non-physical because of finite-size sampling.

    The probability of an inconclusive outcome cannot be smaller than the overlap
    p_ = |<0|alpha>|^2. If a sampled probability falls below it, it is set to
    p_ + 1e-3 and the matching conclusive outcome is renormalised.

    Parameters
    ----------
    p : list
        List whose first element p[0] is a dict of probabilities p[b][x]
        (b = outcome, x = input). Modified in place.
    alpha : float
        Amplitude of the coherent state sent for x = 1.
    impl : str
        Implementation: '1' (time bin) corrects only x = 1,
        '2' (BPSK) corrects both x = 0 and x = 1.

    Returns
    -------
    list or None
        The corrected `p`, or None if `impl` is neither '1' nor '2'.
    """
    if impl not in ('1', '2'):
        return None

    psi0 = qt.coherent(20, 0)
    psi1 = qt.coherent(20, alpha)
    p_ = abs(psi0.dag() * psi1)**2  # minimal inconclusive probability

    if p[0][2][1] < p_:
        p[0][2][1] = p_ + 1e-3
        p[0][1][1] = 1 - (p_ + 1e-3)

    if impl == '2' and p[0][2][0] < p_:
        p[0][2][0] = p_ + 1e-3
        p[0][0][0] = 1 - (p_ + 1e-3)

    return p
