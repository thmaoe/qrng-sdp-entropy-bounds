"""Lower bound on the Shannon (von Neumann) entropy with the BFF method (Brown-Fawzi-Fawzi).

impl = 0: normal protocol; impl = 1: Bob also has an input y.
"""

import cvxpy as cp
import numpy as np

from common import get_cm, get_quadrature, get_rhos


def getMatricesFaster(xs, asize, bs, impl=0, ys=2):
    """Create the cvxpy variables of the BFF SDP.

    Parameters
    ----------
    xs : int
        Number of inputs x.
    asize : int
        Number of values of the variable Eve guesses.
    bs : int
        Number of outcomes b.
    impl : int
        0: no input for Bob; otherwise Bob has an input y.
    ys : int
        Number of inputs y (only if impl != 0).

    Returns
    -------
    Mbs : dict
        Bob's POVM elements Mbs[b] (or Mbs[b][y]), 2x2 Hermitian (only 2 states, so dimension 2).
    Xis, Thetas : dict
        2x2 complex variables Xis[b][x][a] and Thetas[b][x][a]
        (or Xis[b][y][x][a], Thetas[b][y][x][a]).
    """
    if impl == 0:
        Mbs = {b: cp.Variable((2, 2), hermitian=True) for b in range(bs)}
        Xis = {b: {x: {a: cp.Variable((2, 2), complex=True) for a in range(asize)}
                   for x in range(xs)} for b in range(bs)}
        Thetas = {b: {x: {a: cp.Variable((2, 2), complex=True) for a in range(asize)}
                      for x in range(xs)} for b in range(bs)}

    else:  # Bob has an input y
        Mbs = {b: {y: cp.Variable((2, 2), hermitian=True) for y in range(ys)} for b in range(bs)}
        Xis = {b: {y: {x: {a: cp.Variable((2, 2), complex=True) for a in range(asize)}
                       for x in range(xs)} for y in range(ys)} for b in range(bs)}
        Thetas = {b: {y: {x: {a: cp.Variable((2, 2), complex=True) for a in range(asize)}
                          for x in range(xs)} for y in range(ys)} for b in range(bs)}

    return Mbs, Xis, Thetas


def getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, asize, bs, impl=0, ys=2):
    """Constraints of the BFF SDP.

    Parameters
    ----------
    Mbs, Xis, Thetas : dict
        Variables returned by `getMatricesFaster`.
    rho : dict
        States sent, rho[x].
    p : dict
        Observed probabilities p[b][x] (or p[b][x][y] if impl != 0).
    xs, asize, bs, impl, ys :
        See `getMatricesFaster`.

    Returns
    -------
    list
        cvxpy constraints.
    """
    constraints = []

    if impl == 0:
        for a in range(asize):
            for x in range(xs):
                sumb_xi = 0.0
                sumb_theta = 0.0
                for bb in range(bs):
                    G = cp.bmat([[Mbs[bb],       Xis[bb][x][a]],
                                 [Xis[bb][x][a], Thetas[bb][x][a]]])
                    constraints += [G >> 0.0]

                    sumb_xi += Xis[bb][x][a]
                    sumb_theta += Thetas[bb][x][a]

                constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(2) / float(2)]
                constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(2) / float(2)]

        # POVM completeness
        sumb_m = 0.0
        for b in range(bs):
            sumb_m += Mbs[b]
        constraints += [sumb_m == np.eye(2)]

        # Reproduce the observed statistics
        for b in range(bs):
            for x in range(xs):
                constraints += [cp.trace(Mbs[b] @ rho[x]) == p[b][x]]

    else:
        for a in range(asize):
            for x in range(xs):
                for y in range(ys):
                    sumb_xi = 0.0
                    sumb_theta = 0.0
                    for bb in range(bs):
                        G = cp.bmat([[Mbs[bb][y],       Xis[bb][y][x][a]],
                                     [Xis[bb][y][x][a], Thetas[bb][y][x][a]]])
                        constraints += [G >> 0.0]

                        sumb_xi += Xis[bb][y][x][a]
                        sumb_theta += Thetas[bb][y][x][a]

                    constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(2) / float(2)]
                    constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(2) / float(2)]

        # POVM completeness for each y
        for y in range(ys):
            sumb_m = 0.0
            for b in range(bs):
                sumb_m += Mbs[b][y]
            constraints += [sumb_m == np.eye(2)]

        # Reproduce the observed statistics
        for b in range(bs):
            for x in range(xs):
                for y in range(ys):
                    constraints += [cp.trace(Mbs[b][y] @ rho[x]) == p[b][x][y]]

    return constraints


def getHFaster(m, xs, asize, bs, p, rho, w, t, px, impl=0, ys=2):
    """Solve the m-1 BFF SDPs and sum them into the Shannon-entropy lower bound.

    Parameters
    ----------
    m : int
        Number of quadrature nodes.
    xs, asize, bs, impl, ys :
        See `getMatricesFaster`. With asize = 2 (impl = 0 only), Eve guesses
        c = conclusive (b in {0, 1}) / inconclusive (b = 2); otherwise she guesses b.
    p : dict
        Observed probabilities p[b][x] (or p[b][x][y] if impl != 0).
    rho : dict
        States sent, rho[x].
    w, t : array
        Quadrature weights and nodes.
    px : float
        Probability of input x = 1.
    impl: 0 or 1
        wether Bob has input y or not
    ys: int
        number of inputs for Bob, only when impl=1

    Returns
    -------
    float
        Lower bound on the entropy.
    """
    obj = 0.0

    if impl == 0:
        for i in range(m-1):
            Mbs, Xis, Thetas = getMatricesFaster(xs, asize, bs)
            constraints = getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, asize, bs)
            sumb = 0.0
            for a in range(asize):
                sumbb0 = 0.0
                sumbb1 = 0.0
                for bb in range(bs):
                    sumbb0 += Thetas[bb][0][a]
                    sumbb1 += Thetas[bb][1][a]

                if asize == 2:
                    # Coarse-grain b into c: a = 0 <-> b in {0, 1}, a = 1 <-> b = 2
                    if a == 0:
                        Xisa0a = Xis[0][0][a] + Xis[1][0][a]
                        Thetaa0a = Thetas[0][0][a] + Thetas[1][0][a]
                        Xisa1a = Xis[0][1][a] + Xis[1][1][a]
                        Thetaa1a = Thetas[0][1][a] + Thetas[1][1][a]
                    else:
                        Xisa0a = Xis[2][0][a]
                        Thetaa0a = Thetas[2][0][a]
                        Xisa1a = Xis[2][1][a]
                        Thetaa1a = Thetas[2][1][a]

                    sumb += (1-px) * cp.real(cp.trace(rho[0] @ (2*Xisa0a + (1 - t[i]) * Thetaa0a + t[i]*sumbb0)))
                    sumb += px * cp.real(cp.trace(rho[1] @ (2*Xisa1a + (1 - t[i]) * Thetaa1a + t[i]*sumbb1)))

                else:
                    sumb += (1-px) * cp.real(cp.trace(rho[0] @ (2*Xis[a][0][a] + (1 - t[i]) * Thetas[a][0][a] + t[i]*sumbb0)))
                    sumb += px * cp.real(cp.trace(rho[1] @ (2*Xis[a][1][a] + (1 - t[i]) * Thetas[a][1][a] + t[i]*sumbb1)))

            subObj = sumb * (w[i]/(t[i]*np.log(2)))

            prob = cp.Problem(cp.Minimize(subObj), constraints)
            prob.solve(solver='MOSEK')
            obj += prob.value

    else:
        py = [1/2, 1/2]
        for i in range(m-1):
            Mbs, Xis, Thetas = getMatricesFaster(xs, asize, bs, impl, ys)
            constraints = getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, asize, bs, impl, ys)
            sumb = 0.0
            for a in range(asize):
                for y in range(ys):
                    sumbb0 = 0.0
                    sumbb1 = 0.0
                    for bb in range(bs):
                        sumbb0 += Thetas[bb][y][0][a]
                        sumbb1 += Thetas[bb][y][1][a]

                    sumb += py[y] * (1-px) * cp.real(cp.trace(rho[0] @ (2*Xis[a][y][0][a] + (1 - t[i]) * Thetas[a][y][0][a] + t[i]*sumbb0)))
                    sumb += py[y] * px * cp.real(cp.trace(rho[1] @ (2*Xis[a][y][1][a] + (1 - t[i]) * Thetas[a][y][1][a] + t[i]*sumbb1)))

            subObj = sumb * (w[i]/(t[i]*np.log(2)))

            prob = cp.Problem(cp.Minimize(subObj), constraints)
            prob.solve(solver='MOSEK')
            obj += prob.value

    obj += get_cm(m, t, w)
    return obj


def getHDual(delta, p, px):
    """Dual of the BFF SDP, used for the finite-size analysis (GEAT).

    Parameters
    ----------
    delta : float
        Overlap between the two states sent.
    p : dict
        Observed probabilities p[b][x].
    px : float
        Probability of input x = 1.

    Returns
    -------
    H : float
        Lower bound on the entropy.
    Lambdas : list of dict
        Lagrange multipliers lambdas[b][x], useful for min tradeoff function.
    Rs : list of np.ndarray
        Real part of the matrix Lagrange multiplier, useful for min tradeoff function.
    cm : float
        quadrature constant.
    """
    pxs = [1-px, px]

    bs = len(p)
    xs = len(p[0])

    m, t, w = get_quadrature()
    rho = get_rhos(delta)

    H = 0.0
    Lambdas = []
    Rs = []

    for i in range(m-1):
        ti = t[i]
        taui = (w[i]/(t[i]*np.log(2)))

        Lambda = cp.Variable((2, 2), complex=True)
        Gammas = {a: {x: cp.Variable((2, 2), complex=True) for x in range(xs)} for a in range(bs)}
        Deltas = {a: {x: cp.Variable((2, 2), complex=True) for x in range(xs)} for a in range(bs)}
        lambdasbx = {b: {x: cp.Variable() for x in range(xs)} for b in range(bs)}
        As = {a: {x: {b: cp.Variable((2, 2), complex=True) for b in range(bs)} for x in range(xs)} for a in range(bs)}
        Bs = {a: {x: {b: cp.Variable((2, 2), complex=True) for b in range(bs)} for x in range(xs)} for a in range(bs)}
        Cs = {a: {x: {b: cp.Variable((2, 2), complex=True) for b in range(bs)} for x in range(xs)} for a in range(bs)}

        constraints = []

        # Somehow this one works but not with the delta_ab, maybe because we set 0 which is not good
        constraints += [sum([As[a][x][b] for a in range(bs) for x in range(xs)])
                        == sum([lambdasbx[b][x] * rho[x] for x in range(xs)]) + Lambda for b in range(bs)]

        for a in range(bs):
            for x in range(xs):
                for b in range(bs):
                    Sab10 = 2*taui*(a == b)*pxs[x]*rho[x] + Gammas[a][x] - 1/2*cp.trace(Gammas[a][x])*np.eye(2)
                    Sab11 = taui * ((1-ti)*(a == b) + ti) * pxs[x] * rho[x] + Deltas[a][x] - 1/2*cp.trace(Deltas[a][x])*np.eye(2)
                    constraints += [Cs[a][x][b] == Sab11]
                    constraints += [Bs[a][x][b] + Bs[a][x][b].H == Sab10]

        for a in range(bs):
            for x in range(xs):
                for b in range(bs):
                    Rab = cp.bmat([[As[a][x][b],   Bs[a][x][b]],
                                   [Bs[a][x][b].H, Cs[a][x][b]]])
                    constraints += [Rab >> 0]

        sumbx = sum([lambdasbx[b][x] * p[b][x] for b in range(bs) for x in range(xs)])

        obj = cp.real(-cp.trace(Lambda) - sumbx)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver='MOSEK')

        lambdas = {b: {x: lambdasbx[b][x].value for x in range(xs)} for b in range(bs)}

        H += prob.value
        Lambdas.append(lambdas)
        Rs.append(np.real(Lambda.value))

    cm = get_cm(m, t, w)
    H += cm

    return H, Lambdas, Rs, cm


def runOpti(delta, p, px, asize=2, impl=0, ys=2):
    """Compute the Shannon-entropy lower bound (primal BFF SDP).

    Parameters
    ----------
    delta : float
        Overlap between the two states sent.
    p : dict
        Observed probabilities p[b][x] (or p[b][x][y] if impl != 0).
    px : float
        Probability of input x = 1.
    asize : int
        Number of values Eve guesses: 2 (default) = Eve guesses c (conclusive / inconclusive),
        3 = Eve guesses b.
    impl : int
        0: no input for Bob; 1: Bob has an input y.
    ys : int
        Number of inputs y (only if impl != 0).

    Returns
    -------
    float
        Lower bound on the entropy.
    """
    rho = get_rhos(delta)
    m, t, w = get_quadrature()
    xs = len(p[0])
    bs = len(p)

    return getHFaster(m, xs, asize, bs, p, rho, w, t, px, impl, ys)
