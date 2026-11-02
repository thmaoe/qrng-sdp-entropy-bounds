"""Shannon-entropy lower bound combining a Gram-matrix (moment) relaxation with BFF.

Version with 2 outcomes b and an input y for Bob (p(y) = 1/2). The Gram matrices are
indexed by the 53 operators listed in `genZEqs` (S): id, M_by, Z_bxy, Z_bxy^dag and
their products.

This is still experimental, might contain some bugs.
"""

import itertools
import re

import chaospy
import cvxpy as cp
import numpy as np

N_OPS = 53   # size of the operator list S
TOL = 1e-6   # tolerance used for equality constraints

# Gram-matrix entries <id M_by>, <M_by id>, <M_by M_by> = p(b|x,y), as M_IDS[b][y]
M_IDS = {0: [[(0, 1), (1, 0), (1, 1)], [(0, 2), (2, 0), (2, 2)]],
         1: [[(0, 3), (3, 0), (3, 3)], [(0, 4), (4, 0), (4, 4)]]}


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
        (106, 106) matrix such that the objective is trace(Gamma @ A).
    """
    py = 1/2

    # x = 0 block: <Z_b0y Z_b0y^dag> (S indices 21, 22, 25, 26),
    #              <M_by Z_b0y>, <M_by Z_b0y^dag> (29, 30, 33, 34, 37, 38, 41, 42),
    #              <M_by Z_b0y^dag Z_b0y> (45, 47, 49, 51)
    A = np.zeros((N_OPS, N_OPS))
    for j in [21, 25]:
        A[0, j] = (1-py) * (1-px) * ti * taui
    for j in [22, 26]:
        A[0, j] = (py) * (1-px) * ti * taui
    for j in [29, 30, 37, 38]:
        A[0, j] = (1-py) * (1-px) * taui
    for j in [33, 34, 41, 42]:
        A[0, j] = (py) * (1-px) * taui
    for j in [45, 49]:
        A[0, j] = (1-py) * (1-px) * taui * (1-ti)
    for j in [47, 51]:
        A[0, j] = (py) * (1-px) * taui * (1-ti)

    # x = 1 block: same with the x = 1 operators
    B = np.zeros((N_OPS, N_OPS))
    for j in [23, 27]:
        B[0, j] = (1-py) * (1-px) * ti * taui
    for j in [24, 28]:
        B[0, j] = (py) * (1-px) * ti * taui
    for j in [31, 32, 39, 40]:
        B[0, j] = (1-py) * (1-px) * taui
    for j in [35, 36, 43, 44]:
        B[0, j] = (py) * (1-px) * taui
    for j in [46, 50]:
        B[0, j] = (1-py) * (1-px) * taui * (1-ti)
    for j in [48, 52]:
        B[0, j] = (py) * (1-px) * taui * (1-ti)

    return np.kron(A, np.array([[1, 0], [0, 0]])).T + np.kron(B, np.array([[0, 0], [0, 1]])).T


def getCs(Gamma00, Gamma11, p, constraints):
    """Add the constraints reproducing the observed statistics p(b|x,y).

    Parameters
    ----------
    Gamma00, Gamma11 : cp.Variable
        Gram matrices for x = 0 and x = 1.
    p : dict
        Observed probabilities p[b][x][y].
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    for b in range(2):
        for x in range(2):
            Gamma = Gamma00 if x == 0 else Gamma11
            for y in range(2):
                for i in range(3):
                    C = np.zeros((N_OPS, N_OPS))
                    C[M_IDS[b][y][i]] = 1
                    constraints += [(cp.trace(Gamma @ C.T)) == p[b][x][y]]


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
    """Add the POVM completeness constraints (sum_b M_by = id for each y), same bounds as in `getDs`.

    Parameters
    ----------
    Gammas : list of cp.Variable
        Gram matrices [Gamma00, Gamma01, Gamma10, Gamma11].
    delta : float
        Overlap between the two states sent.
    constraints : list
        List of cvxpy constraints, extended in place.
    """
    for y in range(2):
        combis = getCombs([M_IDS[0][y], M_IDS[1][y]])  # 9 combinations

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
    reduced with: id is neutral, M_by commute with the Z's (moved to the right),
    M_by M_by = M_by, M_by M_b'y = 0 for b != b', and M_by M_b'y' is kept for y != y'.

    Returns
    -------
    zero_list : list of tuple
        Indices (i, j) of entries that are 0.
    equal_list : list of tuple
        Pairs of indices ((i, j), (k, l)) of entries that are equal.
    """
    # Operator list
    S = [
        "id",

        # Mby terms
        "M00", "M01", "M10", "M11",

        # Zbxy and Zbxy† terms (grouped in pairs)
        "Z000", "Z000dag", "Z001", "Z001dag",
        "Z010", "Z010dag", "Z011", "Z011dag",
        "Z100", "Z100dag", "Z101", "Z101dag",
        "Z110", "Z110dag", "Z111", "Z111dag",

        # ZbxyZbxy† terms
        "Z000Z000dag", "Z001Z001dag", "Z010Z010dag", "Z011Z011dag",
        "Z100Z100dag", "Z101Z101dag", "Z110Z110dag", "Z111Z111dag",

        # MbyZbxy and MbyZbxy† terms (grouped in pairs)
        "M00Z000", "M00Z000dag", "M00Z010", "M00Z010dag",
        "M01Z001", "M01Z001dag", "M01Z011", "M01Z011dag",
        "M10Z100", "M10Z100dag", "M10Z110", "M10Z110dag",
        "M11Z101", "M11Z101dag", "M11Z111", "M11Z111dag",

        # MbyZbxy†Zbxy terms
        "M00Z000dagZ000", "M00Z010dagZ010",
        "M01Z001dagZ001", "M01Z011dagZ011",
        "M10Z100dagZ100", "M10Z110dagZ110",
        "M11Z101dagZ101", "M11Z111dagZ111"
    ]

    # Adjoints of the operators in S
    S_ = [
        "id",

        # Mby terms
        "M00", "M01", "M10", "M11",

        # Zbxy† and Zbxy terms (grouped in pairs)
        "Z000dag", "Z000", "Z001dag", "Z001",
        "Z010dag", "Z010", "Z011dag", "Z011",
        "Z100dag", "Z100", "Z101dag", "Z101",
        "Z110dag", "Z110", "Z111dag", "Z111",

        # ZbxyZbxy† terms
        "Z000Z000dag", "Z001Z001dag", "Z010Z010dag", "Z011Z011dag",
        "Z100Z100dag", "Z101Z101dag", "Z110Z110dag", "Z111Z111dag",

        # MbyZbxy† and MbyZbxy terms (grouped in pairs)
        "M00Z000dag", "M00Z000", "M00Z010dag", "M00Z010",
        "M01Z001dag", "M01Z001", "M01Z011dag", "M01Z011",
        "M10Z100dag", "M10Z100", "M10Z110dag", "M10Z110",
        "M11Z101dag", "M11Z101", "M11Z111dag", "M11Z111",

        # MbyZbxy†Zbxy terms
        "M00Z000dagZ000", "M00Z010dagZ010",
        "M01Z001dagZ001", "M01Z011dagZ011",
        "M10Z100dagZ100", "M10Z110dagZ110",
        "M11Z101dagZ101", "M11Z111dagZ111"
    ]

    pattern = re.compile(r"M\d\d|Z\d\d\d(?:dag)?|id")

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
        """Product of the M factors: 'id' if none, M_by if all equal,
        M_by M_b'y' if y != y', None (= 0) otherwise."""
        if not M_factors:
            return 'id'
        if len(set(M_factors)) == 1:
            return M_factors[0]
        elif M_factors[0][2] != M_factors[1][2]:  # different y
            return M_factors[0]+M_factors[1]
        return None

    def canonical_product(op1, op2):
        """Canonical form (Z factors, M factors) of op1*op2, or None if it is 0."""
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
    m_entries = [idx for b in range(2) for group in M_IDS[b] for idx in group]

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
        Observed probabilities p[b][x][y], b, x, y in {0, 1}.
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
    m_in = 2
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
