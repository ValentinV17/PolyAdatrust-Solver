"""
example.py
==========
Runnable demonstration of PolyTR on the Rosenbrock function.

Usage
-----
    python example.py
"""

import numpy as np

from polytr import minimize

def rosenbrock(x):
    """sum_{i} [100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2]."""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x):
    """Gradient of sum_{i} [100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2]."""
    g = np.zeros_like(x)
    g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:]  += 200 * (x[1:] - x[:-1] ** 2)
    return g


if __name__ == "__main__":
    n  = 10
    x0 = np.full(n, -1.0)

    print(f"Minimizing {n}-dimensional Rosenbrock from x0 = {x0}\n")

    result = minimize(
        grad=rosenbrock_grad,
        x0=x0,
        tol=1e-4,
        max_iter=10000,
    )

    # Stop once the objective value drops below a threshold.  The Rosenbrock
    # minimum is f = 0 at x = (1, ..., 1).
    
    # result = minimize(
    #     grad=rosenbrock_grad,
    #     x0=x0,
    #     stopping_criterion=lambda s: rosenbrock(s.x) < 1e-8,
    #     max_iter=10000,
    # )
    
    print("\n")
    print("Optimization results:")
    print(f"Converged : {result['success']}")
    print(f"Iterations: {result['n_iter']}")
    print(f"x*        : {np.round(result['x'], 6)}")
    print(f"||grad||  : {result['gnorm']:.4e}")
    print(f"f(x*)     : {rosenbrock(result['x']):.4e}")
