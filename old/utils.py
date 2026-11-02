## Useful various functions

import qutip as qt

def handle_non_phys(p, alpha, impl = '1'): ##Handling non physical probas which happens bc of finite size sampling
    if impl == '1':
        
        psi0 = qt.coherent(20, 0)
        psi1 = qt.coherent(20, alpha)
        p_ = abs(psi0.dag()*psi1)**2
        if p[0][2][1] < p_:
            p[0][2][1] = p_ + 1e-3
            p[0][1][1] = 1 - (p_+1e-3)

        return p
    elif impl == '2':
        psi0 = qt.coherent(20, 0)
        psi1 = qt.coherent(20, alpha)
        p_ = abs(psi0.dag()*psi1)**2
        if (p[0][2][1])  < p_:
            p[0][2][1] = p_ + 1e-3
            p[0][1][1] = 1 - (p_+1e-3)
        if (p[0][2][0])  < p_:
            p[0][2][0] = p_ + 1e-3
            p[0][0][0] = 1 - (p_+1e-3)

        return p
