##Function to compute lower bound on Shannon entropy
## impl 0 = normal; impl = 1 adding y input to Bob

import cvxpy as cp
import numpy as np
import chaospy

def getMatricesFaster(xs, asize, bs, impl=0, ys=2): #Faster version -- better to use this one

    if impl==0:
        Mbs = {}

        for i in range(bs):
            Mbs[i] = cp.Variable((2,2), hermitian=True) ##we have only 2 inputs so we in dimension 2

        Xis = {}

        for b in range(bs):
            Xis[b] = {}
            for x in range(xs):
                Xis[b][x] = {}
                for a in range(asize):
                    Xis[b][x][a] = cp.Variable((2,2), complex = True)

        Thetas = {}
        for b in range(bs):
            Thetas[b] = {}
            for x in range(xs):
                Thetas[b][x] = {}
                for a in range(asize):
                    Thetas[b][x][a] = cp.Variable((2,2), complex = True) 

        return Mbs, Xis, Thetas
    
    else: #adding y input to Bob
        Mbs = {}

        for i in range(bs):
            Mbs[i] = {}
            for y in range(ys): 
                Mbs[i][y] = cp.Variable((2,2), hermitian=True) ##we have only 2 inputs so we in dimension 2

        Xis = {}

        for b in range(bs):
            Xis[b] = {}
            for y in range(ys):
                Xis[b][y] = {}
                for x in range(xs):
                    Xis[b][y][x] = {}
                    for a in range(asize):
                        Xis[b][y][x][a] = cp.Variable((2,2), complex = True)

        Thetas = {}
        for b in range(bs):
            Thetas[b] = {}
            for y in range(ys):
                Thetas[b][y] = {}
                for x in range(xs):
                    Thetas[b][y][x] = {}
                    for a in range(asize):
                        Thetas[b][y][x][a] = cp.Variable((2,2), complex = True) 

        return Mbs, Xis, Thetas

def getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, asize, bs, impl=0, ys=2): 
    if impl == 0:
        constraints = []

        for a in range(asize):
            for x in range(xs):
                sumb_xi = 0.0
                sumb_theta = 0.0
                for bb in range(bs): #We have Xi_bb^{xb}
                    G = cp.bmat([[Mbs[bb],           Xis[bb][x][a]], 
                                [Xis[bb][x][a], Thetas[bb][x][a]]])
                    constraints += [G >> 0.0] 

                    sumb_xi += Xis[bb][x][a]
                    sumb_theta += Thetas[bb][x][a]
                
                constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(2) / float(2)]
                constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(2) / float(2)]

        sumb_m = 0.0
        for b in range(bs):
            sumb_m += Mbs[b]       
        constraints += [sumb_m == np.eye(2)]

        for b in range(bs):
            for x in range(xs):
                constraints += [cp.trace(Mbs[b] @ rho[x]) == p[b][x]]
        
        return constraints
    
    else:
        constraints = []

        for a in range(asize):
            for x in range(xs):
                for y in range(ys):
                    sumb_xi = 0.0
                    sumb_theta = 0.0
                    for bb in range(bs): 
                        G = cp.bmat([[Mbs[bb][y],           Xis[bb][y][x][a]], 
                                    [Xis[bb][y][x][a], Thetas[bb][y][x][a]]])
                        constraints += [G >> 0.0] 

                        sumb_xi += Xis[bb][y][x][a]
                        sumb_theta += Thetas[bb][y][x][a]
                    
                    constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(2) / float(2)]
                    constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(2) / float(2)]

        for y in range(ys):
            sumb_m = 0.0
            for b in range(bs):
                sumb_m += Mbs[b][y]
            constraints += [sumb_m == np.eye(2)]

        for b in range(bs):
            for x in range(xs):
                for y in range(ys):
                    constraints += [cp.trace(Mbs[b][y] @ rho[x]) == p[b][x][y]]
        
        return constraints
        

def getHFaster(m, xs, asize, bs, p, rho, w, t, px, impl = 0, ys=2):
    if impl == 0:

        obj = 0.0
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

        cm = 0.0

        for i in range(m-1):
            cm += w[i]/(t[i]*np.log(2))

        obj += cm
        return obj

    else:
        py = [1/2, 1/2]
        obj = 0.0
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

        cm = 0.0

        for i in range(m-1):
            cm += w[i]/(t[i]*np.log(2))

        obj += cm
        return obj

def getHDual(delta, p, px):

    px = [1-px, px]

    bs = len(p)
    xs = len(p[0])

    m_in = 4
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
    t = t[0]

    rho0 = np.array([[1.,0],[0,0]])
    rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)], 
                    [delta*np.sqrt(1-delta**2), 1-delta**2]])
    rho1 = rho1 / np.trace(rho1) #bc of floating errors

    rho = {0: rho0, 1: rho1}

    H = 0.0
    Lambdas = []
    Rs = []

    for i in range(m-1):
        ti = t[i]
        taui =  (w[i]/(t[i]*np.log(2)))

        Lambda = cp.Variable((2,2), complex=True)

        Gammas = {}
        for a in range(bs):
            Gammas[a] = {}
            for x in range(xs):
                Gammas[a][x] = cp.Variable((2,2), complex = True)

        Deltas = {}
        for a in range(bs):
            Deltas[a] = {}
            for x in range(xs):
                Deltas[a][x] = cp.Variable((2,2), complex = True)

        lambdasbx = {}
        for b in range(bs):
            lambdasbx[b] = {}
            for x in range(xs):
                lambdasbx[b][x] = cp.Variable()

        As = {}
        for a in range(bs):
            As[a] = {}
            for x in range(xs):
                As[a][x] = {}
                for b in range(bs):
                    As[a][x][b] = cp.Variable((2,2), complex = True)

        Bs = {}
        for a in range(bs):
            Bs[a] = {}
            for x in range(xs):
                Bs[a][x] = {}
                for b in range(bs):
                    Bs[a][x][b] = cp.Variable((2,2), complex = True)

        Cs = {}
        for a in range(bs):
            Cs[a] = {}
            for x in range(xs):
                Cs[a][x] = {}
                for b in range(bs):
                    Cs[a][x][b] = cp.Variable((2,2), complex = True)
        
        constraints = []

        constraints += [sum([ As[a][x][b] for a in range(bs) for x in range(xs)]) == sum([lambdasbx[b][x] * rho[x] for x in range(xs)]) + Lambda for b in range(bs)] ##somehow this one works but not with the delta_ab, maybe bc we set 0 which is not good

        for a in range(bs):
            for x in range(xs):
                for b in range(bs):
                    Sab10 = 2*taui*(a==b)*px[x]*rho[x] + Gammas[a][x] - 1/2*cp.trace(Gammas[a][x])*np.eye(2)
                    Sab11 = taui * ((1-ti)*(a==b) + ti) * px[x] * rho[x] + Deltas[a][x] - 1/2*cp.trace(Deltas[a][x])*np.eye(2)
                    constraints += [Cs[a][x][b] == Sab11]
                    constraints += [Bs[a][x][b] + Bs[a][x][b].H == Sab10]

        for a in range(bs):
            for x in range(xs):
                for b in range(bs):
                    Rab = cp.bmat([[As[a][x][b], Bs[a][x][b]], 
                                [Bs[a][x][b].H, Cs[a][x][b]]])
                    constraints += [Rab >> 0]

        sumbx = sum([lambdasbx[b][x] * p[b][x] for b in range(bs) for x in range(xs)])

        obj = cp.real(-cp.trace(Lambda) - sumbx)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver='MOSEK')
        lambdas = {}
        for b in range(bs):
            lambdas[b] = {}
            for x in range(xs):
                lambdas[b][x] = (lambdasbx[b][x].value)
                
        H += prob.value
        Lambdas.append(lambdas)
        Rs.append(np.real(Lambda.value))
        
    cm = 0.0

    for i in range(m-1):
        cm += w[i]/(t[i]*np.log(2))

    H += cm

    return H, Lambdas, Rs, cm

def runOpti(delta, p, px, asize = 3, impl = 0, ys=2):

    rho0 = np.array([[1.,0],[0,0]])
    rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)], 
                    [delta*np.sqrt(1-delta**2), 1-delta**2]])
    rho1 = rho1 / np.trace(rho1) #bc of floating errors

    rho = {0: rho0, 1: rho1}

    m_in = 4
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=0, upper=1)
    t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
    t = t[0]
    xs = len(p[0])
    bs = len(p)

    return getHFaster(m, xs, asize, bs, p, rho, w, t, px, impl, ys)
    