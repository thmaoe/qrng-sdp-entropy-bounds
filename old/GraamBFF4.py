import numpy as np
import cvxpy as cp
import re 
import itertools
import chaospy

def getA(px, ti, taui):
    py = 1/2
    A = np.zeros((53,53))
    
    indices = [(0,21), (0,22), (0,25), (0,26), (0,29), (0,30), (0,33), (0,34), (0,37), (0,38), (0,41), (0,42), (0,45), (0,47), (0,49), (0,51)]

    for id in indices:
        if id in [(0,21), (0,25)]:
            A[id[0], id[1]] = (1-py) * (1-px) * ti * taui
        elif id in [(0,22), (0,26)]:
            A[id[0], id[1]] = (py) * (1-px) * ti * taui
        elif id in [(0,29), (0,30), (0,37), (0,38)]:
            A[id[0], id[1]] = (1-py) * (1-px) * taui
        elif id in [(0,33), (0,34), (0,41), (0,42)]:
            A[id[0], id[1]] = (py) * (1-px) * taui
        elif id in [(0,45), (0,49)]:
            A[id[0], id[1]] = (1-py) * (1-px) * taui * (1-ti)
        elif id in [(0,47), (0,51)]:
            A[id[0], id[1]] = (py) * (1-px) * taui * (1-ti)
        else:
            print("wrong id")
        

    indices = [(0,23), (0,24), (0,27), (0,28), (0,31), (0,32), (0,35), (0,36), (0,39), (0,40), (0,43), (0,44), (0,46), (0,48), (0,50), (0,52)]

    B = np.zeros((53,53))
    for id in indices:
        if id in [(0,23), (0,27)]:
            B[id[0], id[1]] = (1-py) * (1-px) * ti * taui
        elif id in [(0,24), (0,28)]:
            B[id[0], id[1]] = (py) * (1-px) * ti * taui
        elif id in [(0,31), (0,32), (0,39), (0,40)]:
            B[id[0], id[1]] = (1-py) * (1-px) * taui
        elif id in [(0,35), (0,36), (0,43), (0,44)]:
            B[id[0], id[1]] = (py) * (1-px) * taui
        elif id in [(0,46), (0,50)]:
            B[id[0], id[1]] = (1-py) * (1-px) * taui * (1-ti)
        elif id in [(0,48), (0,52)]:
            B[id[0], id[1]] = (py) * (1-px) * taui * (1-ti)
        else:
            print("wrong id")
    
    return np.kron(A, np.array([[1,0],[0,0]])).T + np.kron(B, np.array([[0,0], [0,1]])).T

def getCs(Gamma00, Gamma11, p, constraints):
    Cs = {b: {x :{y: [np.zeros((53,53)) for _ in range(3)] for y in range(2)} for x in range(2)} for b in range(2)}
    ids = {0: [[(0,1), (1,0), (1,1)], [(0,2), (2,0), (2,2)]], 1: [[(0,3), (3,0), (3,3)], [(0,4), (4,0), (4,4)]]}
    
    for b in range(2):
        for x in range(2):
            for y in range(2):
                Cs[b][x][y][0][ids[b][y][0]] = 1
                Cs[b][x][y][1][ids[b][y][1]] = 1
                Cs[b][x][y][2][ids[b][y][2]] = 1
                if x == 0:
                    for i in range(3):
                        constraints += [(cp.trace(Gamma00 @ Cs[b][x][y][i].T)) == p[b][x][y]]
                elif x == 1:
                    for i in range(3):
                        constraints += [(cp.trace(Gamma11 @ Cs[b][x][y][i].T)) == p[b][x][y]]

def getDs(Gammas, delta, constraints):

    Ds = [np.zeros((53,53)) for _ in range(4)]

    Ds[0][0,0] = 1
    Ds[1][0,0] = 1
    Ds[2][0,0] = 1
    Ds[3][0,0] = 1

    constraints += [(cp.trace(Gammas[0] @ Ds[0].T)) <= 1 + 1e-6]
    constraints += [(cp.trace(Gammas[0] @ Ds[0].T)) >= 1 - 1e-6]
    constraints += [(cp.trace(Gammas[1] @ Ds[1].T)) >= delta]
    constraints += [(cp.trace(Gammas[2] @ Ds[2].T)) >= delta]
    constraints += [(cp.trace(Gammas[3] @ Ds[3].T)) <= 1 + 1e-6]
    constraints += [(cp.trace(Gammas[3] @ Ds[3].T)) >= 1 - 1e-6]

def getCombs(indices):
# All 27 possible combinations of one from each group
    raw_combinations = itertools.product(indices[0], indices[1])

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

    indices = [[(0,1), (1,0), (1,1)], [(0,3), (3,0), (3,3)]]
    combis = getCombs(indices)

    Ms = [[np.zeros((53,53)) for _ in range(len(combis))] for _ in range(4)]

    for i in range(4):
        for j, comb in enumerate(combis):
            Ms[i][j][comb[0]] = 1
            Ms[i][j][comb[1]] = 1

    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) <= 1 + 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) >= 1 - 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[1] @ Ms[1][j].T) >= delta for j in range(len(combis))]
    constraints += [cp.trace(Gammas[2] @ Ms[2][j].T) >= delta for j in range(len(combis))]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) <= 1 + 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) >= 1 - 1e-6 for j in range(len(combis))]

    indices = [[(0,2), (2,0), (2,2)], [(0,4), (4,0), (4,4)]]
    combis = getCombs(indices)

    Ms = [[np.zeros((53,53)) for _ in range(len(combis))] for _ in range(4)]

    for i in range(4):
        for j, comb in enumerate(combis):
            Ms[i][j][comb[0]] = 1
            Ms[i][j][comb[1]] = 1

    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) <= 1 + 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[0] @ Ms[0][j].T) >= 1 - 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[1] @ Ms[1][j].T) >= delta for j in range(len(combis))]
    constraints += [cp.trace(Gammas[2] @ Ms[2][j].T) >= delta for j in range(len(combis))]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) <= 1 + 1e-6 for j in range(len(combis))]
    constraints += [cp.trace(Gammas[3] @ Ms[3][j].T) >= 1 - 1e-6 for j in range(len(combis))]

def genZEqs():

    # Your operator lists
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




    S_ = [
    "id",

    # Mby terms
    "M00", "M01", "M10", "M11",

    # Zbxy and Zbxy† terms (grouped in pairs)
    "Z000dag", "Z000", "Z001dag", "Z001",
    "Z010dag", "Z010", "Z011dag", "Z011",
    "Z100dag", "Z100", "Z101dag", "Z101",
    "Z110dag", "Z110", "Z111dag", "Z111",

    # ZbxyZbxy† terms
    "Z000Z000dag", "Z001Z001dag", "Z010Z010dag", "Z011Z011dag",
    "Z100Z100dag", "Z101Z101dag", "Z110Z110dag", "Z111Z111dag",

    # MbyZbxy and MbyZbxy† terms (grouped in pairs)
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
        elif M_factors[0][2] != M_factors[1][2]:
            return M_factors[0]+M_factors[1]
        else:
            return None

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
            F = np.zeros((53,53))
            F[id] = 1
            constraints += [(cp.trace(Gammas[i] @ F.T)) <= 1e-6, (cp.trace(Gammas[i] @ F.T)) >= -1e-6]

    for id in equal_list:
        if id[0] not in [(0,1), (1,0), (1,1), (0,2), (2,0), (2,2), (0,3), (3,0), (3,3), (0,4), (4,0), (4,4)] and id[1] not in [(0,1), (1,0), (1,1), (0,2), (2,0), (2,2), (0,3), (3,0), (3,3), (0,4), (4,0), (4,4)]:
            for i in range(4):
                F = np.zeros((53,53))
                F[id[0]] = 1
                F[id[1]] = -1 
                constraints += [(cp.trace(Gammas[i] @ F.T)) <= 1e-6, (cp.trace(Gammas[i] @ F.T)) >= -1e-6]


def runSDP(p, delta, px):
    m_in = 2
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
    t = t[0]

    H = 0.0

    for i in range(m-1):

        ti = t[i]
        taui =  (w[i]/(t[i]*np.log(2)))

        Gamma00 = cp.Variable((53,53), complex = False)
        Gamma01 = cp.Variable((53,53), complex = False)
        Gamma10 = cp.Variable((53,53), complex = False)
        Gamma11 = cp.Variable((53,53), complex = False)

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
    