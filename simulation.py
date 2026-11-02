"""Simulation of the different protocols and of their output statistics.

Implementations (`impl`):
    '0' = photon number, '1' = time bin, '2' = BPSK, '3' = modified BPSK.

Probabilities are stored as nested dicts p[b][x] = p(b|x), with outcome
b in {0, 1, 2} and input x in {0, 1}. For impl '0' and '2' only b = 1 and
b = 2 are used, and b = 2 stands for the outcome "0" (no click).
"""

from math import exp

import numpy as np
from scipy.special import erf

TD = 34e-9         # detector dead time [s]
PULSE_RATE = 50e6  # pulse rate [Hz], 50 MHz as in the paper


def _rescale_clicks(p, x, cd):
    """Multiply the click probabilities p[0][x], p[1][x] by `cd` and renormalise p[2][x]. Used for deadtime."""
    p0x = p[0][x] * cd
    p1x = p[1][x] * cd
    p[0][x] = p0x
    p[1][x] = p1x
    p[2][x] = 1 - (p0x + p1x)


def _add_noise(p, pnoise):
    """Mix p[1][x] and p[2][x] with a uniform distribution of weight `pnoise` (in place). For BPSK only."""
    for b in (1, 2):
        for x in (0, 1):
            p[b][x] = (1 - pnoise) * p[b][x] + pnoise * 1/2


def _get_delta(alpha, impl):
    """Overlap delta between the two states sent, for a given amplitude and implementation."""
    if impl == '0':
        return exp(-abs(alpha)**2/2)
    elif impl == '1':
        return exp(-abs(alpha)**2)
    else:
        return exp(-2*abs(alpha)**2)


def measure(p, eff=1, impl='0'):
    """Sample the outcome b of a single measurement.

    Parameters
    ----------
    p : list of float
        Probabilities [p(0|x), p(1|x), p(2|x)] for the input x of this round.
    eff : float
        Detector efficiency (only used for impl '0' and '1').
    impl : str
        Implementation ('0', '1', '2' or '3').

    Returns
    -------
    int
        Outcome b in {0, 1, 2}. For impl '0' and '2', 2 means "0" (no click).
    """
    if impl == '0':
        if np.random.rand() < p[1]:
            if np.random.rand() < eff:
                return 1
            return 2  # detector missed the photon
        return 2

    elif impl == '1':
        r = np.random.rand()     # for the measurement
        reff = np.random.rand()  # for the efficiency

        if r < p[0]:
            return 0 if reff < eff else 2
        elif r < p[0] + p[1]:
            return 1 if reff < eff else 2
        return 2

    elif impl == '2':
        if np.random.rand() < p[1]:
            return 1
        return 2

    elif impl == '3':
        r = np.random.rand()
        if r < p[0]:
            return 0
        elif r < p[0] + p[1]:
            return 1
        return 2


def pick_x(px1):
    """Sample the input x: 1 with probability `px1`, 0 otherwise."""
    if np.random.rand() < px1:
        return 1
    return 0


def get_stat(data, deadtime=False):
    """Estimate p(b|x) from a list of measurement results, i.e frequencies.

    Parameters
    ----------
    data : list of tuple
        Results (b, x) of each round.
    deadtime : bool
        If True, apply the detector dead-time correction from the paper:
        click probabilities are multiplied by 1 / (1 + TD * click_rate).

    Returns
    -------
    p : dict
        Estimated probabilities p[b][x].
    counts : dict
        Raw counts counts[b][x].
    """
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    total = {0: 0, 1: 0}

    for b, x in data:
        counts[b][x] += 1
        total[x] += 1

    p = {b: {x: (counts[b][x] / total[x]) if total[x] > 0 else 0 for x in (0, 1)}
         for b in (0, 1, 2)}

    for x in (0, 1):
        N_total = total[x]
        N_click = counts[0][x] + counts[1][x]

        if N_total == 0:
            continue

        click_rate = (N_click / N_total) * PULSE_RATE
        cd = 1 / (1 + TD * click_rate) if deadtime else 1

        _rescale_clicks(p, x, cd)

    return p, counts


def getProbas(alpha, eff, dc, impl, deadtime=False, d=0):
    """Theoretical probabilities p(b|x) for a given implementation.

    Parameters
    ----------
    alpha : float
        Amplitude of the coherent state(s).
    eff : float
        Detector efficiency (impl '0' and '1'), modelled as a beam splitter of
        transmissivity eff: |alpha> -> |alpha*sqrt(eff)>.
    dc : float
        Dark-count probability (impl '0' and '1').
    impl : str
        Implementation ('0', '1', '2' or '3').
    deadtime : bool
        If True, apply the detector dead-time correction from the paper.
    d : float
        Threshold of the inconclusive region (impl '3' only).

    Returns
    -------
    delta : float
        Overlap between the two states sent.
    p : dict
        Probabilities p[b][x].
    """
    if impl == '0':
        noclick = exp(-abs(alpha)**2*eff)  # |<0|alpha*sqrt(eff)>|^2
        p10 = dc
        p00 = 1 - dc
        p11 = 1 - (1 - dc)*noclick
        p01 = (1 - dc)*noclick  # no dark count * no click
        delta = exp(-abs(alpha)**2/2)
        # Here 2 = "0": getting no click is the inconclusive outcome
        p = {0: {0: 0.0, 1: 0.0}, 1: {0: p10, 1: p11}, 2: {0: p00, 1: p01}}

    elif impl == '1':
        noclick = exp(-abs(alpha)**2*eff)
        p00 = 1 - (1 - dc)*noclick
        p10 = (1 - p00)*dc  # no click in first bin * dark count
        p11 = 1 - (1 - dc)*noclick
        p01 = (1 - p11)*dc
        p20 = (1 - dc)**2*noclick
        p21 = (1 - dc)**2*noclick
        p = {0: {0: p00, 1: p01}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}
        delta = exp(-abs(alpha)**2)

    elif impl == '2':
        s = np.sqrt(2) * np.abs(alpha)
        p00 = 1/2 * (1 + erf(s))
        p01 = 1/2 * (1 - erf(s))
        p10 = 1/2 * (1 - erf(s))
        p11 = 1/2 * (1 + erf(s))
        delta = np.exp(-2*(np.abs(alpha)**2))
        # Here 2 = "0", as for impl '0'
        p = {0: {0: 0, 1: 0}, 1: {0: p10, 1: p11}, 2: {0: p00, 1: p01}}

    elif impl == '3':
        s = np.sqrt(2) * np.abs(alpha)
        p00 = 1/2 * (1 - erf(d - s))
        p01 = 1/2 * (1 - erf(d + s))
        p10 = 1/2 * (1 - erf(d + s))
        p11 = 1/2 * (1 - erf(d - s))
        p20 = 1/2 * (erf(d - s) + erf(d + s))
        p21 = 1/2 * (erf(d - s) + erf(d + s))
        delta = np.exp(-2*(np.abs(alpha)**2))
        p = {0: {0: p00, 1: p01}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}

    if deadtime:  # technique from the paper
        for x in (0, 1):
            click_rate = (p[0][x] + p[1][x]) * PULSE_RATE
            cd = 1 / (1 + TD * click_rate)
            _rescale_clicks(p, x, cd)

    return delta, p


def doSimul(alpha, px1=1/2, impl='0', nPoints=100000, eff=1, deadtime=False, badSource=False, d=0, pnoise=0):
    """Simulate `nPoints` rounds of the protocol and return the statistics.

    Parameters
    ----------
    alpha : float
        Amplitude of the coherent state(s) (real).
    px1 : float
        Probability of choosing input x = 1.
    impl : str
        Implementation ('0', '1', '2' or '3').
    nPoints : int
        Number of rounds.
    eff : float
        Detector efficiency (applied when sampling, impl '0' and '1').
    deadtime : bool
        If True, apply the dead-time correction when computing the statistics.
    badSource : bool
        If True, the source is imperfect: each round sends alpha + N(0, 0.05/(2*alpha))
        and delta is computed from the 99.9th percentile of |alpha|.
    d : float
        Threshold of the inconclusive region (impl '3' only).
    pnoise : float
        Model for BPSK as in Davide's paper. Only usable for BPSK (impl '2')

    Returns
    -------
    (delta, stats) or (alphamax, delta, stats) if `badSource`
        alphamax : 99.9th percentile of the sampled |alpha|.
        delta : overlap between the two states.
        stats : (p, counts) as returned by `get_stat`.
    """
    if badSource:
        goodAlpha = alpha

    data = []
    alphas = []
    _, p = getProbas(alpha, 1, 0, impl, False, d)  # dead time is applied when collecting the stats
    if pnoise != 0:
        _add_noise(p, pnoise)

    for _ in range(nPoints):
        if badSource:
            if alpha == 0:
                alphas.append(alpha)
            else:
                # the source is not perfect so it sends |alpha + delta>
                alpha = goodAlpha + np.random.normal(0, 0.05/(2*goodAlpha))
                alphas.append(alpha)
            _, p = getProbas(alpha, 1, 0, impl, False, d)
            if pnoise != 0:
                _add_noise(p, pnoise)

        x = pick_x(px1)
        p_ = [p[b][x] for b in range(3)]
        b = measure(p_, eff, impl)
        data.append((b, x))

    if badSource:
        alphamax = np.percentile(np.abs(alphas), 99.9)
        delta = _get_delta(alphamax, impl)
    else:
        delta = _get_delta(alpha, impl)

    stats = get_stat(data, deadtime)

    if badSource:
        return alphamax, delta, stats
    return delta, stats
