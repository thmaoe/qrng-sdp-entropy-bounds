### simulation.py contains code to generate statistics for various versions of the protocol.
- impl = '0' -> photon number, '1' -> time bin, '2' -> bpsk, '3' -> modified bpsk
- measure(p, eff, impl): gives result b of one round of the experiment given probas p(b|x), efficiency of detector eff and implementation used.
- get_stat(data, deadtime): gives the statistics for multiple rounds of experiment, given data a list of results (b,x) and possibility to apply the deadtime of detector.
- getProbas(alpha, eff, dc, impl, deadtime, d): gives theoretical probas for various implementations
- doSimul(alpha, px1, impl, nPoints, eff, deadtime, badSource, d): gives stats for nPoints round of experiment, with wanted parameters.

### shannonLower.py contains code to compute lower bound on Shannon entropy
- runOpti(delta, p, px, asize = 3, impl = 0, ys=2): allows to compute the lower bound given wanted parameters. impl is wether we include (=1) or not (=0) an input y to Bob. Can change wether we consider the b or c variable with asize.
- getHDual(delta, p, px): allows to compute the dual problem, to be used in finite size. It returns H the lower bound, Lambdas and Rs the Lagrange multipliers and cm, the constant from BFF. 

### HminLower.py contains code to compute the lower bound on Hmin
- just use getHmin(p, delta, px), if want to consider 3 outputs (b instead of c), use the one from HminLowerb.py

### ShannonEAT.py containts code to compute lower bound on smooth min entropy with EAT
- getH3(ps, N, lambdas, Rs, cm, nB=3): gives the EAT lower bound. It needs a list of probas p(b|x), the number of points, Lagrange multiplier from the Dual and the BFF constant.

### GraamBFF4.py containts code to compute Shannon lower bound using combination of Graam and BFF.
- runSDP(p, delta, px): gives this lower bound. Can modify the m parameter from BFF directly in this function.

