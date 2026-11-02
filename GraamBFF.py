"""Shannon-entropy lower bound combining a Gram-matrix (moment) relaxation with BFF.

Version with 3 outcomes b and no input for Bob. The Gram matrices are indexed by the
40 operators listed in `genZEqs` (S): id, M_b, Z_bx, Z_bx^dag and their products.

This is still experimental, might contain some bugs.
""" 

import itertools
import re

import chaospy
import cvxpy as cp
import numpy as np

N_OPS = 40   # size of the operator list S
TOL = 1e-8   # tolerance used for equality constraints

# Gram-matrix entries <id M_b>, <M_b id>, <M_b M_b> = p(b|x), for b = 0, 1, 2
M_IDS = [[(0, 1), (1, 0), (1, 1)], [(0, 2), (2, 0), (2, 2)], [(0, 3), (3, 0), (3, 3)]]


def getA(px, ti, taui):
    """Objective matrix of the BFF SDP for the quadrature node (ti, taui).

    Parameters
    ----------
    px : float
        Probability of input x = 1.
    ti : float
        Quadrature node t_i.
    taui : float
        Quadrature coefficient w_i / (t_i ln 2).

    Returns
    -------
    np.ndarray
        (80, 80) matrix such that the objective is trace(Gamma @ A).
    """
    # x = 0 block: <M_b Z_b0>, <M_b Z_b0^dag> (S indices 22-27),
    #              <M_b Z_b0^dag Z_b0> (34-36), <Z_b0 Z_b0^dag> (16-18)
    A = np.zeros((N_OPS, N_OPS))
    for j in [22, 23, 24, 25, 26, 27]:
        A[0, j] = (1-px) * taui
    for j in [34, 35, 36]:
        A[0, j] = (1-px) * taui * (1-ti)
    for j in [16, 17, 18]:
        A[0, j] = (1-px) * taui * (ti)

    # x = 1 block: same with S indices 28-33, 37-39 and 19-21
    B = np.zeros((N_OPS, N_OPS))
    for j in [28, 29, 30, 31, 32, 33]:
        B[0, j] = (px) * taui
    for j in [37, 38, 39]:
        B[0, j] = (px) * taui * (1-ti)
    for j in [19, 20, 21]:
        B[0, j] = (px) * taui * (ti)

    return np.kron(A, np.array([[1, 0], [0, 0]])).T + np.kron(B, np.array([[0, 0], [0, 1]])).T


def getCs(Gamma00, Gamma11, p, constraints):
    """Add the constraints reproducing the observed statistics p(b|x).

    Parameters
    ----------
    Gamma00, Gamma11 : cp.Variable
        Gram matrices for x = 0 and x = 1.
    p : dict
        Observed probabilities p[b][x].
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    for b in range(3):
        for x in range(2):
            Gamma = Gamma00 if x == 0 else Gamma11
            for i in range(3):
                C = np.zeros((N_OPS, N_OPS))
                C[M_IDS[b][i]] = 1
                constraints += [(cp.trace(Gamma @ C.T)) == p[b][x]]


def getDs(Gammas, delta, constraints):
    """Add the normalisation (<id> = 1) and overlap (<id> >= delta) constraints.

    Parameters
    ----------
    Gammas : list of cp.Variable
        Gram matrices [Gamma00, Gamma01, Gamma10, Gamma11].
    delta : float
        Overlap between the two states sent.
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    D = np.zeros((N_OPS, N_OPS))
    D[0, 0] = 1

    constraints += [(cp.trace(Gammas[0] @ D.T)) <= 1 + TOL]
    constraints += [(cp.trace(Gammas[0] @ D.T)) >= 1 - TOL]
    constraints += [(cp.trace(Gammas[1] @ D.T)) >= delta]
    constraints += [(cp.trace(Gammas[2] @ D.T)) >= delta]
    constraints += [(cp.trace(Gammas[3] @ D.T)) <= 1 + TOL]
    constraints += [(cp.trace(Gammas[3] @ D.T)) >= 1 - TOL]


def getCombs(indices):
    """All combinations taking one element from each group, ignoring their order.

    Parameters
    ----------
    indices : list of list
        Groups of Gram-matrix indices.

    Returns
    -------
    list of tuple
        Unique (sorted) combinations.
    """
    raw_combinations = itertools.product(*indices)
    # Sort each combination so that permutations become identical, then remove duplicates
    return list({tuple(sorted(combo)) for combo in raw_combinations})


def getMs(Gammas, delta, constraints):
    """Add the POVM completeness constraints (sum_b M_b = id), same bounds as in `getDs`.

    Parameters
    ----------
    Gammas : list of cp.Variable
        Gram matrices [Gamma00, Gamma01, Gamma10, Gamma11].
    delta : float
        Overlap between the two states sent.
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    combis = getCombs(M_IDS)  # 27 combinations

    Ms = []
    for comb in combis:
        M = np.zeros((N_OPS, N_OPS))
        for idx in comb:
            M[idx] = 1
        Ms.append(M)

    constraints += [cp.trace(Gammas[0] @ M.T) <= 1 + TOL for M in Ms]
    constraints += [cp.trace(Gammas[0] @ M.T) >= 1 - TOL for M in Ms]
    constraints += [cp.trace(Gammas[1] @ M.T) >= delta for M in Ms]
    constraints += [cp.trace(Gammas[2] @ M.T) >= delta for M in Ms]
    constraints += [cp.trace(Gammas[3] @ M.T) <= 1 + TOL for M in Ms]
    constraints += [cp.trace(Gammas[3] @ M.T) >= 1 - TOL for M in Ms]


def genZEqs():
    """Find the Gram-matrix entries that vanish or are equal by operator algebra.

    Entry (i, j) is <S_[i] S[j]>, where S_ is the list of adjoints of S. Products are
    reduced with: id is neutral, M_b commute with the Z's (moved to the right),
    M_b M_b = M_b and M_b M_b' = 0 for b != b'.

    Returns
    -------
    zero_list : list of tuple
        Indices (i, j) of entries that are 0.
    equal_list : list of tuple
        Pairs of indices ((i, j), (k, l)) of entries that are equal.
    """
    # Operator list
    S = [
        "id", "M0", "M1", "M2",
        "Z00", "Z00dag", "Z10", "Z10dag", "Z20", "Z20dag",
        "Z01", "Z01dag", "Z11", "Z11dag", "Z21", "Z21dag",
        "Z00Z00dag", "Z10Z10dag", "Z20Z20dag", "Z01Z01dag",
        "Z11Z11dag", "Z21Z21dag",
        "M0Z00", "M0Z00dag", "M1Z10", "M1Z10dag", "M2Z20",
        "M2Z20dag", "M0Z01", "M0Z01dag", "M1Z11", "M1Z11dag",
        "M2Z21", "M2Z21dag", "M0Z00dagZ00", "M1Z10dagZ10",
        "M2Z20dagZ20", "M0Z01dagZ01", "M1Z11dagZ11", "M2Z21dagZ21"
    ]

    # Adjoints of the operators in S
    S_ = [
        "id", "M0", "M1", "M2",
        "Z00dag", "Z00", "Z10dag", "Z10", "Z20dag", "Z20",
        "Z01dag", "Z01", "Z11dag", "Z11", "Z21dag", "Z21",
        "Z00Z00dag", "Z10Z10dag", "Z20Z20dag", "Z01Z01dag",
        "Z11Z11dag", "Z21Z21dag",
        "M0Z00dag", "M0Z00", "M1Z10dag", "M1Z10", "M2Z20dag",
        "M2Z20", "M0Z01dag", "M0Z01", "M1Z11dag", "M1Z11",
        "M2Z21dag", "M2Z21", "M0Z00dagZ00", "M1Z10dagZ10",
        "M2Z20dagZ20", "M0Z01dagZ01", "M1Z11dagZ11", "M2Z21dagZ21"
    ]

    pattern = re.compile(r"M\d|Z\d\d(?:dag)?|id")

    def split_factors(op):
        return pattern.findall(op)

    def move_Ms_to_right(factors):
        factors = factors[:]
        changed = True
        while changed:
            changed = False
            for i in range(len(factors)-1):
                if factors[i].startswith('M') and factors[i+1].startswith('Z'):
                    factors[i], factors[i+1] = factors[i+1], factors[i]
                    changed = True
        return factors

    def simplify_Ms(M_factors):
        """Product of the M factors: 'id' if none, M_b if all equal, None (= 0) otherwise."""
        if not M_factors:
            return 'id'
        if len(set(M_factors)) == 1:
            return M_factors[0]
        return None

    def canonical_product(op1, op2):
        """Canonical form (Z factors, M factor) of op1*op2, or None if it is 0."""
        all_factors = split_factors(op1) + split_factors(op2)
        all_factors = [f for f in all_factors if f != 'id']  # id is neutral
        all_factors = move_Ms_to_right(all_factors)

        M_factors = [f for f in all_factors if f.startswith('M')]
        Z_factors = [f for f in all_factors if not f.startswith('M')]

        M_simpl = simplify_Ms(M_factors)
        if M_simpl is None:
            return None

        return (tuple(Z_factors), M_simpl)

    product_map = {}
    zero_list = []
    for i, op1 in enumerate(S_):
        for j, op2 in enumerate(S):
            cform = canonical_product(op1, op2)
            if cform is None:
                zero_list.append((i, j))
            else:
                product_map[(i, j)] = cform

    equal_list = []
    keys = list(product_map.keys())
    for idx1 in range(len(keys)):
        for idx2 in range(idx1+1, len(keys)):
            key1 = keys[idx1]
            key2 = keys[idx2]
            if product_map[key1] == product_map[key2]:
                equal_list.append((key1, key2))

    return zero_list, equal_list


def getZEqs(Gammas, constraints):
    """Add the constraints from `genZEqs` (zero and equal entries) to every Gram matrix.

    Pairs involving the entries used in `getCs` are skipped.

    Parameters
    ----------
    Gammas : list of cp.Variable
        Gram matrices [Gamma00, Gamma01, Gamma10, Gamma11].
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    zero_list, equal_list = genZEqs()
    m_entries = [idx for group in M_IDS for idx in group]

    for idx in zero_list:
        for i in range(4):
            F = np.zeros((N_OPS, N_OPS))
            F[idx] = 1
            constraints += [(cp.trace(Gammas[i] @ F.T)) <= TOL, (cp.trace(Gammas[i] @ F.T)) >= -TOL]

    for idx in equal_list:
        if idx[0] not in m_entries and idx[1] not in m_entries:
            for i in range(4):
                F = np.zeros((N_OPS, N_OPS))
                F[idx[0]] = 1
                F[idx[1]] = -1
                constraints += [(cp.trace(Gammas[i] @ F.T)) <= TOL, (cp.trace(Gammas[i] @ F.T)) >= -TOL]


def runSDP(p, delta, px):
    """Shannon-entropy lower bound from the Gram + BFF SDPs.

    Parameters
    ----------
    p : dict
        Observed probabilities p[b][x], b in {0, 1, 2}, x in {0, 1}.
    delta : float
        Overlap between the two states sent.
    px : float
        Probability of input x = 1.

    Returns
    -------
    float
        Lower bound on the entropy.
    """
    # Gauss-Radau quadrature; m_in sets the BFF parameter m = 2*m_in
    m_in = 4
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in, distribution, 1.0)
    t = t[0]

    H = 0.0

    for i in range(m-1):
        ti = t[i]
        taui = (w[i]/(t[i]*np.log(2)))

        # Blocks <psi_x| . |psi_x'> of the full Gram matrix
        Gamma00 = cp.Variable((N_OPS, N_OPS), complex=False)
        Gamma01 = cp.Variable((N_OPS, N_OPS), complex=False)
        Gamma10 = cp.Variable((N_OPS, N_OPS), complex=False)
        Gamma11 = cp.Variable((N_OPS, N_OPS), complex=False)

        Gammas = [Gamma00, Gamma01, Gamma10, Gamma11]

        Gamma = (cp.kron(Gamma00, np.array([[1, 0], [0, 0]])) + cp.kron(Gamma01, np.array([[0, 1], [0, 0]]))
                 + cp.kron(Gamma10, np.array([[0, 0], [1, 0]])) + cp.kron(Gamma11, np.array([[0, 0], [0, 1]])))

        constraints = [Gamma >> 0]

        getCs(Gamma00, Gamma11, p, constraints)
        getDs(Gammas, delta, constraints)
        getMs(Gammas, delta, constraints)
        getZEqs(Gammas, constraints)

        A = getA(px, ti, taui)

        obj = (cp.trace(Gamma @ A))

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver='MOSEK', verbose=False)

        H += prob.value

    cm = 0.0
    for i in range(m-1):
        cm += w[i]/(t[i]*np.log(2))

    H += cm

    return H
