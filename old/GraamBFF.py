import numpy as np
import cvxpy as cp
import re 
import itertools
import chaospy

def getA(px, ti, taui):
    A = np.zeros((40,40))
    
    indices = [(0,22), (0,23), (0,34), (0,16), (0,24), (0,25), (0,35), (0,17), (0,26), (0,27), (0,36), (0,18)]

    for id in indices:
        if id in [(0,22), (0,23), (0,24), (0,25), (0,26), (0,27)]:
            A[id[0], id[1]] = (1-px) * taui
        elif id in [(0,34), (0,35), (0,36)]:
            A[id[0], id[1]] = (1-px) * taui * (1-ti)
        else:
            A[id[0], id[1]] = (1-px) * taui * (ti)

    indices = [(0,28), (0,29), (0,37), (0,19), (0,30), (0,31), (0,38), (0,20), (0,32), (0,33), (0,39), (0,21)]

    B = np.zeros((40,40))
    for id in indices: 
        if id in [(0,28), (0,29), (0,30), (0,31), (0,32), (0,33)]:
            B[id[0], id[1]] = (px) * taui
        elif id in [(0,37), (0,38), (0,39)]:
            B[id[0], id[1]] = (px) * taui * (1-ti)
        else:
            B[id[0], id[1]] = (px) * taui * (ti)
    
    return np.kron(A, np.array([[1,0],[0,0]])).T + np.kron(B, np.array([[0,0], [0,1]])).T

def getCs(Gamma00, Gamma11, p, constraints):
    Cs = {b:{x :[np.zeros((40,40)) for _ in range(3)] for x in range(2)} for b in range(3)}
    ids = {0: [(0,1), (1,0), (1,1)], 1: [(0,2), (2,0), (2,2)], 2: [(0,3), (3,0), (3,3)]}
    
    for b in range(3):
        for x in range(2):
            Cs[b][x][0][ids[b][0]] = 1
            Cs[b][x][1][ids[b][1]] = 1
            Cs[b][x][2][ids[b][2]] = 1
            if x == 0:
                for i in range(3):
                    constraints += [(cp.trace(Gamma00 @ Cs[b][x][i].T)) == p[b][x]]
            elif x == 1:
                for i in range(3):
                    constraints += [(cp.trace(Gamma11 @ Cs[b][x][i].T)) == p[b][x]]

def getDs(Gammas, delta, constraints):

    Ds = [np.zeros((40,40)) for _ in range(4)]

    Ds[0][0,0] = 1
    Ds[1][0,0] = 1
    Ds[2][0,0] = 1
    Ds[3][0,0] = 1

    constraints += [(cp.trace(Gammas[0] @ Ds[0].T)) <= 1 + 1e-8]
    constraints += [(cp.trace(Gammas[0] @ Ds[0].T)) >= 1 - 1e-8]
    constraints += [(cp.trace(Gammas[1] @ Ds[1].T)) >= delta]
    constraints += [(cp.trace(Gammas[2] @ Ds[2].T)) >= delta]
    constraints += [(cp.trace(Gammas[3] @ Ds[3].T)) <= 1 + 1e-8]
    constraints += [(cp.trace(Gammas[3] @ Ds[3].T)) >= 1 - 1e-8]

def getCombs(indices):
# All 27 possible combinations of one from each group
    raw_combinations = itertools.product(indices[0], indices[1], indices[2])

    # Use sets to remove permutations that are just re-ordered versions of each other
    unique_sets = set()

    for combo in raw_combinations:
        # Sort to make permutations identical
        sorted_combo = tuple(sorted(combo))
        unique_sets.add(sorted_combo)

    # Convert to list if needed
    unique_combinations = list(unique_sets)

    return unique_combinations

def getMs(Gammas, delta, constraints):
    Ms = [[np.zeros((40,40)) for _ in range(27)] for _ in range(4)]

    indices = [[(0,1), (1,0), (1,1)], [(0,2), (2,0), (2,2)], [(0,3), (3,0), (3,3)]]
    combis = getCombs(indices)

    for i in range(4):
        for j, comb in enumerate(combis):
            Ms[i][j][comb[0]] = 1
            Ms[i][j][comb[1]] = 1
            Ms[i][j][comb[2]] = 1

    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) <= 1 + 1e-8 for j in range(27)]
    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) >= 1 - 1e-8 for j in range(27)]
    constraints += [cp.trace(Gammas[1] @ Ms[1][j].T) >= delta for j in range(27)]
    constraints += [cp.trace(Gammas[2] @ Ms[2][j].T) >= delta for j in range(27)]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) <= 1 + 1e-8 for j in range(27)]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) >= 1 - 1e-8 for j in range(27)]

def genZEqs():

    # Your operator lists
    S = [
        "id", "M0", "M1", "M2",
        "Z00", "Z00dag", "Z10", "Z10dag", "Z20", "Z20dag", 
        "Z01", "Z01dag", "Z11", "Z11dag", "Z21", "Z21dag", 
        "Z00Z00dag", "Z10Z10dag","Z20Z20dag", "Z01Z01dag", 
        "Z11Z11dag", "Z21Z21dag",
        "M0Z00", "M0Z00dag", "M1Z10", "M1Z10dag", "M2Z20", 
        "M2Z20dag", "M0Z01", "M0Z01dag", "M1Z11", "M1Z11dag",
        "M2Z21", "M2Z21dag", "M0Z00dagZ00", "M1Z10dagZ10", 
        "M2Z20dagZ20", "M0Z01dagZ01", "M1Z11dagZ11", "M2Z21dagZ21"
    ]

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
                if factors[i].startswith('M') and (factors[i+1].startswith('Z')):
                    factors[i], factors[i+1] = factors[i+1], factors[i]
                    changed = True
        return factors

    def simplify_Ms(M_factors):
        if not M_factors:
            return 'id'
        unique = set(M_factors)
        if len(unique) == 1:
            return M_factors[0]
        return None  # zero

    def canonical_product(op1, op2):
        factors1 = split_factors(op1)
        factors2 = split_factors(op2)
        all_factors = factors1 + factors2
        
        # Remove all 'id' factors (neutral element)
        all_factors = [f for f in all_factors if f != 'id']
        
        # Move all Ms to the right
        all_factors = move_Ms_to_right(all_factors)
        
        # Separate Ms and Z factors
        M_factors = [f for f in all_factors if f.startswith('M')]
        Z_factors = [f for f in all_factors if not f.startswith('M')]
        
        # Simplify Ms
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
                zero_list.append((i,j))
            else:
                product_map[(i,j)] = cform

    equal_list = []
    checked_pairs = set()
    keys = list(product_map.keys())

    for idx1 in range(len(keys)):
        for idx2 in range(idx1+1, len(keys)):
            key1 = keys[idx1]
            key2 = keys[idx2]
            if product_map[key1] == product_map[key2]:
                if (key1, key2) not in checked_pairs and (key2, key1) not in checked_pairs:
                    equal_list.append((key1, key2))
                    checked_pairs.add((key1, key2))

    return zero_list, equal_list


def getZEqs(Gammas, constraints):
    zero_list, equal_list = genZEqs()

    for id in zero_list:
        for i in range(4):
            F = np.zeros((40,40))
            F[id] = 1
            constraints += [(cp.trace(Gammas[i] @ F.T)) <= 1e-8, (cp.trace(Gammas[i] @ F.T)) >= -1e-8]

    for id in equal_list:
        if id[0] not in [(0,1), (1,0), (1,1), (0,2), (2,0), (2,2), (0,3), (3,0), (3,3)] and id[1] not in [(0,1), (1,0), (1,1), (0,2), (2,0), (2,2), (0,3), (3,0), (3,3)]:
            for i in range(4):
                F = np.zeros((40,40))
                F[id[0]] = 1
                F[id[1]] = -1 
                constraints += [(cp.trace(Gammas[i] @ F.T)) <= 1e-8, (cp.trace(Gammas[i] @ F.T)) >= -1e-8]


def runSDP(p, delta, px):
    m_in = 4
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
    t = t[0]

    H = 0.0

    for i in range(m-1):

        ti = t[i]
        taui =  (w[i]/(t[i]*np.log(2)))

        Gamma00 = cp.Variable((40,40), complex = False)
        Gamma01 = cp.Variable((40,40), complex = False)
        Gamma10 = cp.Variable((40,40), complex = False)
        Gamma11 = cp.Variable((40,40), complex = False)

        Gammas = [Gamma00, Gamma01, Gamma10, Gamma11]

        Gamma = cp.kron(Gamma00, np.array([[1,0],[0,0]])) + cp.kron(Gamma01, np.array([[0,1],[0,0]])) + cp.kron(Gamma10, np.array([[0,0],[1,0]])) + cp.kron(Gamma11, np.array([[0,0],[0,1]]))

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
    