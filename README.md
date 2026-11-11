# xxxx

*still need to update that with paper. Also will need to add names etc. Also add thesis for details on supp things.*

- asymptotic bounds on the **Shannon entropy** (BFF method) and on the **min-entropy** H_min (SDPs),
- finite-size bounds on the **smooth min-entropy** with the **GEAT** (Generalized Entropy Accumulation Theorem).

## Notebooks

| Notebook | Content |
|---|---|
| `results_paper.ipynb` | Reproduces the figures and results of the paper: photon-number and BPSK encodings (ideal case, bad detector / source, noise model, block-size study, finite size, real data). |
| `results.ipynb` | Additional results not in the paper: time-bin encoding, photon number, BPSK with and without inconclusive outcome (choice of `d`), comparison of the `c` and `b` variables, and the Gram + BFF bounds from `GraamBFF.py` (**experimental**). |

## Installation

Python with `numpy`, `scipy`, `cvxpy`, `chaospy`, `matplotlib` (and `qutip` for `utils.py`).
All SDPs are solved with **MOSEK**, which needs a license (free for academics).

## Conventions

- **Encodings** (`impl` in `simulation.py`): `'0'` photon number, `'1'` time bin, `'2'` BPSK, `'3'` BPSK with inconclusive outcome.
- **Probabilities** are nested dicts `p[b][x] = p(b|x)`, with input `x ∈ {0, 1}` and outcome `b ∈ {0, 1, 2}`:
  - for `'1'` and `'3'`, `b = 2` is the inconclusive outcome;
  - for `'0'` and `'2'`, only `b = 1` and `b = 2` occur, and `b = 2` stands for the outcome "0" (no click).
- **Variable Eve guesses:** `b` (the outcome) or `c` (conclusive / inconclusive).
- **`impl` in `shannonLower.py`** means something else: whether Bob has an extra input `y` (`1`) or not (`0`).

## Modules

### `simulation.py`: simulated statistics

- `getProbas(alpha, eff, dc, impl, deadtime=False, d=0)`: theoretical `delta` and `p(b|x)` for an encoding, with detector efficiency `eff`, dark counts `dc`, optional detector dead time and threshold `d` (encoding `'3'`).
- `doSimul(alpha, px1=1/2, impl='0', nPoints=100000, eff=1, deadtime=False, badSource=False, d=0, pnoise=0)`: simulates `nPoints` rounds and returns `(delta, (p, counts))`, or `(alphamax, delta, (p, counts))` with `badSource=True`. `pnoise` (noise model) is only meant for BPSK (`'2'`).
- Lower-level helpers: `measure(p)` samples one outcome (detector efficiency must already be included in `p`, see `getProbas`), `pick_x(px1)` samples the input, `get_stat(data, px, nPoints, deadtime=False)` estimates `p(b|x)` (frequencies) from a list of `(b, x)`, using the fixed `p(x)` given in `px` rather than the empirical one, as in (D15).

### `shannonLower.py`: Shannon-entropy lower bound (BFF)

- `runOpti(delta, p, px, asize=2, impl=0, ys=2)`: asymptotic lower bound. `asize=2`: Eve guesses `c` (default); `asize=3`: Eve guesses `b`. With `impl=1`, Bob has an input `y` (with `ys` values) and `p` is `p[b][x][y]`.
- `getHDual(delta, p, px)`: dual problem, used for finite size. Returns the bound `H`, the Lagrange multipliers `Lambdas` and `Rs`, and the constant `cm`.

### `HminLower.py`: min-entropy lower bound

- `getHmin(p, delta, px, guess_b=False)`: asymptotic lower bound on H_min. Eve guesses `c` by default, `b` with `guess_b=True`. Only for the case without input `y` for Bob.
- `getHminGEAT(p, delta, px, guess_b=False)`: same SDP, returns the guessing probability and the dual variables needed by `ShannonGEAT.getH3min`.

### `ShannonGEAT.py`: finite-size bounds with the GEAT

- `Hgeat(ps, delta, nPoints, pct, route, px, hsrc=0, verbose=False)`: full wrapped function giving the GEAT lower bound. `ps` is a list of distributions (blocks of `nPoints` rounds), the first one being used as reference. `route` is `"bff"` or `"hmin"`, `pct` is the percentage of the min-tradeoff function to use, `px` is the probability that `x = 1`, and `hsrc` is the entropy penalty from the source (classical fluctuations). With `verbose=True`, prints how many blocks were used and which ones.
- `getfBFF(p, lambdas, Rs, cm, hsrc=0)`: min-tradeoff function for the BFF route, from the multipliers returned by `shannonLower.getHDual`.
- `getfHmin(p, mus, w, hsrc=0)`: min-tradeoff function for the H_min route, from the dual variables returned by `HminLower.getHminGEAT` (`w` is the point at which -log2 is linearised, typically the guessing probability).
- `getH3(ps, N, lambdas, Rs, cm, f_ref, pct=0.95, hsrc=0, nB=2)`: GEAT lower bound with the BFF route, for `N` rounds, over the list of distributions `ps`. `f_ref` is the reference min-tradeoff value and `pct` the percentage of it used for `h_th`.
- `getH3min(ps, N, mus, w, f_ref, pct=0.95, hsrc=0, nB=2)`: same with the H_min route.

### `common.py`: shared helpers

- `get_rhos(delta)`: the two states sent, with overlap `delta`.
- `get_quadrature(m_in=4)` and `get_cm(m, t, w)`: Gauss-Radau quadrature and constant `c_m` of the BFF method.

### `utils.py`

- `handle_non_phys(p, alpha, impl='1')`: corrects probabilities made non-physical by finite-size sampling.

### `GraamBFF.py`, `GraamBFF4.py`: Gram + BFF bounds (experimental)

Shannon-entropy lower bound combining a Gram-matrix (moment) relaxation with BFF.
**Still experimental and may contain bugs. See some examples in results.ipynb**

- `GraamBFF.runSDP(p, delta, px)`: 3 outcomes, no input for Bob (used in `results.ipynb`).
- `GraamBFF4.runSDP(p, delta, px)`: 2 outcomes and an input `y` for Bob, with `p[b][x][y]`.
- The BFF parameter `m` is set directly inside `runSDP` (`m_in`).

## Data

`EXP_DATA/` contains the experimental data (counts and conditional probabilities for several mean photon numbers).
