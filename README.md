# PolyTR

**A Polynomial-decay adaptive
Trust-Region Method without
Function Evaluations.**

PolyTR is a gradient-only optimizer for smooth, unconstrained problems `min f(x)`, `x ∈ Rⁿ`. Every iteration uses **one gradient evaluation and zero function evaluations**: there is no acceptance test and no line search. The trust-region radius is set automatically from the gradient norm, and is allowed to grow whenever the gradient passes a polynomially-tightening threshold.

```
.
├── polytr.py          # the solver: minimize() and the trust-region machinery
├── example.py         # runnable demo on the Rosenbrock function
├── requirements.txt   # numpy (numba optional)
└── README.md
```

## Installation

```bash
git clone https://github.com/ValentinV17/PolyTR-Solver.git
cd PolyTR-Solver
pip install -r requirements.txt   # installs numpy; numba is optional and commented out by default
python example.py
```

To use the solver in your own project, copy `polytr.py` next to your code and `from polytr import minimize`. If Numba is installed, the inner kernels are JIT-compiled automatically; otherwise the code falls back to a pure-NumPy path with identical logic.

## Quick start

```python
import numpy as np
from polytr import minimize

# A gradient oracle: grad(x) -> ndarray of the same shape as x
def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:]  +=  200 * (x[1:] - x[:-1] ** 2)
    return g

result = minimize(rosenbrock_grad, np.full(10, -1.0), tol=1e-4)
print(result["success"], result["n_iter"], result["gnorm"])
```

You only ever supply a gradient — there is no `fun=...` argument, by design.

## API

```python
result = minimize(grad, x0, tol=1e-4, max_iter=10000,
                  stopping_criterion=None, verbose=True, **solver_kwargs)
```

`minimize` returns a `dict` with:

| Key | Meaning |
|-----|---------|
| `x` | solution found |
| `gnorm` | gradient norm at the solution |
| `n_iter` | number of iterations (= gradient evaluations) taken |
| `success` | `True` if the stopping criterion was met before `max_iter` |

**Custom stopping criteria.** By default the solver stops when `||grad f(xₖ)|| < tol`. You can pass any rule instead; it is called as `stopping_criterion(solver)` after each step and should return `True` to stop. The solver object exposes:

- `solver.x` — current iterate
- `solver.g` — current gradient
- `solver.gnorm` — `||g||`. 

When `stopping_criterion` is provided, `tol` is ignored.
 
The rule may look at anything it likes, including the objective value. Minimizing with PolyTR never requires evaluating `f`, but a *stopping* check is free to evaluate it on the current iterate `s.x` — so a function-value criterion stays available even though the solver itself is gradient-only:
 
```python
result = minimize(grad, x0, stopping_criterion=lambda s: rosenbrock(s.x) < 1e-8)
```
 
This already proposed in a commented out part of `example.py`, stopping on the Rosenbrock objective value.


**Hyperparameters** (passed as keyword arguments):

| Parameter | Default | Role |
|-----------|---------|------|
| `theta` | `1.10` | polynomial-decay exponent θ, controlling when the radius is allowed to grow |
| `eta` | `0.85` | radius scale η in `Δₖ = η·‖∇f(xₖ)‖ / bₖ` |
| `bmin` | `0.85·1e-4` | lower bound on the scaling factor |

The defaults were selected via an Optuna search in the master's thesis on the problems of the MGH test set and generalize reasonably well on our selection of 152 unconstrained problems got through PyCUTEst , but the optimal values are problem-class dependent — tuning `theta` and `eta` for your problem family is worthwhile.

## How it works

At each iteration PolyTR sets the radius from the gradient norm, takes a Steihaug–Toint CG step inside it, accepts it unconditionally, then updates the scaling factor: it shrinks (radius grows) when the gradient meets the polynomial threshold `‖∇f(xₖ₊₁)‖ ≤ ‖∇f(x₀)‖ / (cₖ+1)^θ`, and otherwise grows via an AdaGrad-Norm accumulator. The model Hessian is maintained with BFGS. Under standard assumptions it reaches `‖∇f(xₖ)‖ ≤ ε` in `O(ε^(−(2 + 1/θ)))` iterations. Full derivation and proofs are in the related master's thesis.

## Contact

Valentin Vauchel — Student of École Polytechnique de Louvain, UCLouvain

valentin.vauchel@gmail.com

