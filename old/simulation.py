## Functions for simulating different protocols and output statistics
## 0 = photon number; 1 = time bin; 2 = bpsk; 3 = modified bpsk

from math import exp
import numpy as np
from scipy.special import erf as erf

def measure(p, eff=1, impl='0'): #depending on given probas (p), gives the result of one measurement, b.
    if impl == '0':
        if np.random.rand() < p[1]:
            if np.random.rand() < eff:
                return 1
            else:
                return 2 #here we return 2 for convennience, but it means 0
        else:
            return 2
                 
    elif impl == '1':

        r = np.random.rand() #for measurement
        reff = np.random.rand() #for efficiency

        if r < p[0]:
            if reff < eff:
                return 0
            else:
                return 2
        elif r < p[0] + p[1]:
            if reff < eff:
                return 1
            else:
                return 2
            
        else:
            return 2     
             
    elif impl == '2':

        if np.random.rand() < p[1]:
            return 1
        else:
            return 2

    elif impl == '3':
        r = np.random.rand()

        if r < p[0]:
            return 0
        elif r < p[0] + p[1]:
            return 1
        else:
            return 2    
        

def pick_x(px1):
    if np.random.rand() < px1:
        return 1
    else:
        return 0


def get_stat(data, deadtime=False): #gives statistics given data (b,x)
    td = 34e-9  # detector dead time in seconds
    pulse_rate = 50e6  # 50 MHz as in paper

    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    total = {0: 0, 1: 0}

    for b, x in data:
        counts[b][x] += 1
        total[x] += 1

    p = {
        b: {
            x: (counts[b][x] / total[x]) if total[x] > 0 else 0
            for x in (0, 1)
        }
        for b in (0, 1, 2)
    }

    for x in (0, 1):
        N_total = total[x]
        N_click = counts[0][x] + counts[1][x]

        if N_total == 0:
            continue

        click_rate = (N_click / N_total) * pulse_rate

        if deadtime: #uses technic from paper
            cd = 1 / (1 + td * click_rate)
        else: 
            cd = 1

        p0x = p[0][x] * cd
        p1x = p[1][x] * cd
        p2x = 1 - (p0x + p1x)  # normalize

        p[0][x] = p0x
        p[1][x] = p1x
        p[2][x] = p2x

    return p, counts

def getProbas(alpha, eff, dc, impl, deadtime = False, d = 0): #theoretical p(b|x)
    if impl == '0':
        p10 = dc
        p00 = 1 - dc
        p11 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p01 = (1 - dc)*exp(-abs(alpha)**2*eff) ##prob no dc * get no click (|<0|alpha>|^2), and eff = BS of transmissivity eta so |alpha> becomes |alpha*sqrt(eta)>
        delta = exp(-abs(alpha)**2/2)
        p = {0: {0: 0.0, 1: 0.0}, 1: {0: p10, 1: p11}, 2: {0: p00, 1: p01}} ##here 2 = inconclusive = 0: when we get nothing it's inconclusive

    elif impl == '1':
        p00 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p10 = (1 - p00)*dc #no click in first * dark count
        p11 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p01 = (1 - p11)*dc
        p20 = (1 - dc)**2*exp(-abs(alpha)**2*eff)
        p21 = (1 - dc)**2*exp(-abs(alpha)**2*eff)
        p = {0: {0: p00, 1: p01}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}
        delta = exp(-abs(alpha)**2)
    
    elif impl == '2':

        p00 = 1/2 * (1 + erf(np.sqrt(2) * np.abs(alpha)))
        p01 = 1/2 * (1 - erf(np.sqrt(2) * np.abs(alpha)))

        p10 = 1/2 * (1 - erf(np.sqrt(2) * np.abs(alpha)))
        p11 = 1/2 * (1 + erf(np.sqrt(2) * np.abs(alpha)))

        delta = np.exp(-2*(np.abs(alpha)**2))

        p = {0: {0: 0, 1: 0}, 1: {0: p10, 1: p11}, 2: {0: p00, 1: p01}}
    
    elif impl == '3':
        
        p00 = 1/2 * (1 - erf(d - np.sqrt(2) * np.abs(alpha)))
        p01 = 1/2 * (1 - erf(d + np.sqrt(2) * np.abs(alpha)))

        p10 = 1/2 * (1 - erf(d + np.sqrt(2) * np.abs(alpha)))
        p11 = 1/2 * (1 - erf(d - np.sqrt(2) * np.abs(alpha)))

        p20 = 1/2 * (erf(d - np.sqrt(2) * np.abs(alpha)) + erf(d + np.sqrt(2) * np.abs(alpha)))
        p21 = 1/2 * (erf(d - np.sqrt(2) * np.abs(alpha)) + erf(d + np.sqrt(2) * np.abs(alpha)))

        delta = np.exp(-2*(np.abs(alpha)**2))

        p = {0: {0: p00, 1: p01}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}
    
    if deadtime: #uses technic from paper
        td = 34e-9  # detector dead time in seconds
        pulse_rate = 50e6  # pulse rate in Hz

        for x in (0, 1):
            p0x = p[0][x]
            p1x = p[1][x]

            click_rate = (p0x + p1x) * pulse_rate

            cd = 1 / (1 + td * click_rate)

            p0x_corr = p0x * cd
            p1x_corr = p1x * cd
            p2x_corr = 1 - (p0x_corr + p1x_corr)  # renormalize

            p[0][x] = p0x_corr
            p[1][x] = p1x_corr
            p[2][x] = p2x_corr

    return delta, p


def doSimul(alpha, px1=1/2, impl='0', nPoints=100000, eff=1, deadtime=False, badSource = False, d=0, pnoise=0):

    if badSource:
        goodAlpha = alpha

    data = []
    alphas = []
    _, p = getProbas(alpha, 1, 0, impl, False, d) #will apply deadtime thing when collecting the stats.
    if pnoise != 0:
        p[1][0] = (1-pnoise)*p[1][0] + pnoise*1/2
        p[1][1] = (1-pnoise)*p[1][1] + pnoise*1/2
        p[2][0] = (1-pnoise)*p[2][0] + pnoise*1/2
        p[2][1] = (1-pnoise)*p[2][1] + pnoise*1/2

    for _ in range(nPoints):
        if badSource:
            if alpha==0:
                alphas.append(alpha)
                #_, p = getProbas(alpha, 1, 0, impl, False, d)
            else:
                alpha = goodAlpha + np.random.normal(0, 0.05/(2*goodAlpha)) #the source is not perfect so sends |alpha + delta>
                alphas.append(alpha)
            _, p = getProbas(alpha, 1, 0, impl, False, d)
            if pnoise != 0:
                p[1][0] = (1-pnoise)*p[1][0] + pnoise*1/2
                p[1][1] = (1-pnoise)*p[1][1] + pnoise*1/2
                p[2][0] = (1-pnoise)*p[2][0] + pnoise*1/2
                p[2][1] = (1-pnoise)*p[2][1] + pnoise*1/2
        else:
            pass

        x = pick_x(px1)
        p_ = [p[b][x] for b in range(3)]
        b = measure(p_, eff, impl)
        data.append((b,x))

    if impl == '0' : 
        if badSource:
            alphamax = np.percentile(np.abs(alphas), 99.9)
            delta = exp(-abs(alphamax)**2/2)
        else:
            delta = exp(-abs(alpha)**2/2)

    elif impl == '1': 
        if badSource:
            alphamax = np.percentile(np.abs(alphas), 99.9)
            delta = exp(-abs(alphamax)**2)
        else:
            delta = exp(-abs(alpha)**2)

    else:
        if badSource:
            alphamax = np.percentile(np.abs(alphas), 99.9)
            delta = exp(-2*abs(alphamax)**2)
        else:
            delta = exp(-2*abs(alpha)**2)

    stats = get_stat(data, deadtime)

    if badSource:
        return alphamax, delta, stats
    else:
        return delta, stats

    
