## Functions to get lower bound on hmin.

import cvxpy as cp
import numpy as np

def getHmin(p, delta, px): ##this works only for implementation 1

    H00 = cp.Variable((2,2), hermitian = True)
    H01 = cp.Variable((2,2), hermitian = True)
    H10 = cp.Variable((2,2), hermitian = True)
    H11 = cp.Variable((2,2), hermitian = True)

    muinc0 = cp.Variable(1)
    muinc1 = cp.Variable(1)
    mu00 = cp.Variable(1)
    mu01 = cp.Variable(1)
    mu10 = cp.Variable(1)
    mu11 = cp.Variable(1)

    rho0 = np.array([[1.,0],[0,0]])
    rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)], [delta*np.sqrt(1-delta**2), 1-delta**2]])
    rho1 = rho1 / np.trace(rho1)

    constraints = [(rho0 * ((1-px) - muinc0) + rho1 * (px - muinc1) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=inc
    constraints += [(rho0 * ((1-px) - muinc0) - rho1 * (muinc1) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(-rho0 * (muinc0) + rho1 * (px - muinc1) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(-rho0 * (muinc0) - rho1 * (muinc1) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]

    constraints += [(-rho0 * (mu00) - rho1 * (mu01) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=0
    constraints += [(-rho0 * (mu00) + rho1 * (px - mu01) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu00) - rho1 * (mu01) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu00) + rho1 * (px - mu01) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]

    constraints += [(-rho0 * (mu10) - rho1 * (mu11) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=1
    constraints += [(-rho0 * (mu10) + rho1 * (px - mu11) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu10) - rho1 * (mu11) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu10) + rho1 * (px - mu11) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]


    obj = cp.real((muinc0 * p[2][0] + muinc1 * p[2][1] + mu10 * p[1][0] + mu11 * p[1][1] + mu01 * p[0][1] + mu00 * p[0][0]))

    obj = cp.real(obj)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver = "MOSEK")

    return -np.log2(prob.value)

def getHminGEAT(p, delta, px): ##this works only for implementation 1

    H00 = cp.Variable((2,2), hermitian = True)
    H01 = cp.Variable((2,2), hermitian = True)
    H10 = cp.Variable((2,2), hermitian = True)
    H11 = cp.Variable((2,2), hermitian = True)

    muinc0 = cp.Variable(1)
    muinc1 = cp.Variable(1)
    mu00 = cp.Variable(1)
    mu01 = cp.Variable(1)
    mu10 = cp.Variable(1)
    mu11 = cp.Variable(1)

    rho0 = np.array([[1.,0],[0,0]])
    rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)], [delta*np.sqrt(1-delta**2), 1-delta**2]])
    rho1 = rho1 / np.trace(rho1)

    constraints = [(rho0 * ((1-px) - muinc0) + rho1 * (px - muinc1) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=inc
    constraints += [(rho0 * ((1-px) - muinc0) - rho1 * (muinc1) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(-rho0 * (muinc0) + rho1 * (px - muinc1) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(-rho0 * (muinc0) - rho1 * (muinc1) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]

    constraints += [(-rho0 * (mu00) - rho1 * (mu01) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=0
    constraints += [(-rho0 * (mu00) + rho1 * (px - mu01) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu00) - rho1 * (mu01) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu00) + rho1 * (px - mu01) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]

    constraints += [(-rho0 * (mu10) - rho1 * (mu11) + H00 - 0.5*cp.trace(H00)*np.eye(2)) << 0] #b=1
    constraints += [(-rho0 * (mu10) + rho1 * (px - mu11) + H01 - 0.5*cp.trace(H01)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu10) - rho1 * (mu11) + H10 - 0.5*cp.trace(H10)*np.eye(2)) << 0]
    constraints += [(rho0 * ((1-px) - mu10) + rho1 * (px - mu11) + H11 - 0.5*cp.trace(H11)*np.eye(2)) << 0]


    obj = cp.real((muinc0 * p[2][0] + muinc1 * p[2][1] + mu10 * p[1][0] + mu11 * p[1][1] + mu01 * p[0][1] + mu00 * p[0][0]))

    obj = cp.real(obj)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver = "MOSEK")

    return prob.value, {2: {0: muinc0.value, 1: muinc1.value}, 1: {0: mu10.value, 1: mu11.value}, 0: {0: mu00.value, 1: mu01.value}}