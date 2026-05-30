"""
PolyTR_example.py
==========
Minimal self-contained example of PolyTR — a gradient-only adaptive
trust-region optimizer.  Only numpy is required (numba is optional).

Quick start
-----------
    python PolyTR_example.py

API
---
    result = minimize(grad, x0)
    result = minimize(grad, x0, tol=1e-8, max_iter=5000, theta=1.3)
    result = minimize(grad, x0, stopping_criterion=lambda s: s.gnorm < 1e-6)
"""

import numpy as np

# Optional Numba JIT for the inner hot path.  Falls back to NumPy silently.
try:
    import numba

    @numba.njit(cache=True)
    def _solve_tau(z, d, Delta):
        a = d @ d
        if a <= 0.0:
            return 0.0
        b    = 2.0 * (z @ d)
        c    = z @ z - Delta * Delta
        disc = max(0.0, b * b - 4.0 * a * c)
        return (-b + disc ** 0.5) / (2.0 * a)

    @numba.njit(cache=True)
    def _bfgs_update(B, s, y):
        Bs   = B @ s
        sTBs = s @ Bs
        if sTBs <= 0.0:
            return
        sTy = s @ y
        n   = len(s)
        for i in range(n):
            for j in range(n):
                B[i, j] += y[i] * y[j] / sTy - Bs[i] * Bs[j] / sTBs

    @numba.njit(cache=True)
    def _steihaug(g, B, Delta, tol):
        n = len(g)
        z = np.zeros(n)
        r = g.copy()
        d = -r.copy()
        for _ in range(n):
            Bd  = B @ d
            dBd = d @ Bd
            if dBd <= 0.0:
                return z + _solve_tau(z, d, Delta) * d
            rTr = r @ r
            if rTr <= 0.0:
                return z
            alpha = rTr / dBd
            z_new = z + alpha * d
            if np.linalg.norm(z_new) >= Delta:
                return z + _solve_tau(z, d, Delta) * d
            r_new = r + alpha * Bd
            if np.linalg.norm(r_new) < tol:
                return z_new
            beta = (r_new @ r_new) / rTr
            d    = -r_new + beta * d
            z, r = z_new, r_new
        norm_z = np.linalg.norm(z)
        return z if norm_z > 0.0 else -g * min(1.0, Delta / (np.linalg.norm(g) + 1e-12))

except ImportError:
    # Pure-NumPy fallback — identical logic, no JIT
    def _solve_tau(z, d, Delta):
        a = d @ d
        if a <= 0:
            return 0.0
        b    = 2 * (z @ d)
        c    = z @ z - Delta ** 2
        disc = max(0.0, b * b - 4 * a * c)
        tau  = (-b + np.sqrt(disc)) / (2 * a)
        return tau if np.isfinite(tau) else 0.0

    def _bfgs_update(B, s, y):
        Bs   = B @ s
        sTBs = s @ Bs
        if sTBs <= 0:
            return
        sTy = s @ y
        B += np.outer(y, y) / sTy - np.outer(Bs, Bs) / sTBs

    def _steihaug(g, B, Delta, tol):
        z = np.zeros_like(g)
        r = g.copy()
        d = -r
        for _ in range(len(g)):
            Bd  = B @ d
            dBd = d @ Bd
            if not np.isfinite(dBd) or dBd <= 0:
                return z + _solve_tau(z, d, Delta) * d
            rTr = r @ r
            if not np.isfinite(rTr) or rTr <= 0:
                return z
            alpha = rTr / dBd
            z_new = z + alpha * d
            if np.linalg.norm(z_new) >= Delta:
                return z + _solve_tau(z, d, Delta) * d
            r_new = r + alpha * Bd
            if np.linalg.norm(r_new) < tol:
                return z_new
            beta  = (r_new @ r_new) / rTr
            d     = -r_new + beta * d
            z, r  = z_new, r_new
        norm_z = np.linalg.norm(z)
        return z if norm_z > 0 else -g * min(1.0, Delta / (np.linalg.norm(g) + 1e-12))


class _BFGS:
    def __init__(self, n):
        self.B = np.eye(n)

    def update(self, s, y):
        if s @ y > 0:
            _bfgs_update(self.B, s, y)

    def matvec(self, v):
        return self.B @ v


class _PolyTR:
    """
    PolyTR: gradient-only adaptive trust-region method.

    At each step the trust-region radius Delta_k = eta * ||g_k|| / b_k is set
    automatically; no function evaluations are required.  The decrease
    threshold tightens polynomially: ||g_k|| <= ||g_0|| / (c_k + 1)^theta,
    where c_k counts how many times the threshold was met.

    Parameters
    ----------
    grad  : callable  —  gradient oracle
    x0    : ndarray   —  starting point
    theta : float     —  polynomial decay exponent  (default 0.85)
    eta   : float     —  trust-region radius scale  (default 1.1)
    bmin  : float     —  minimum scaling factor  (default 1e-4)
    """
    def __init__(self, grad, x0, theta=1.1, eta=0.85, bmin=1e-4*0.85):
        self.grad  = grad
        self.x     = np.asarray(x0, dtype=float).copy()
        self.theta = theta
        self.eta   = eta
        self.bmin  = bmin
        self.ck    = 0.0
        self.g         = grad(self.x)
        self.gnorm     = np.linalg.norm(self.g)
        self.gnorm0    = float(self.gnorm)
        self.gnorm_min = float(self.gnorm)
        self.b         = max(bmin, self.gnorm * eta)
        self.bhat_max  = float(self.gnorm*eta)
        self.H         = _BFGS(len(self.x))

    def step(self):
        Delta     = (self.eta * self.gnorm) / self.b
        d         = _steihaug(self.g, self.H.B, Delta, tol=1e-10)
        x_new     = self.x + d
        g_new     = self.grad(x_new)
        gnorm_new = np.linalg.norm(g_new)
        d_norm    = np.linalg.norm(d)
        if gnorm_new <= self.gnorm0 / (self.ck + 1) ** self.theta:
            bhat  = min(self.bhat_max,
                        max(self.bmin, self.b / 2)
                        if d_norm > 0.5 * Delta
                        else self.b)
            b_new  = bhat
            self.ck += 1
        else:
            b_new = np.sqrt(self.b ** 2 + gnorm_new ** 2)
        self.H.update(d, g_new - self.g)
        self.x         = x_new
        self.g         = g_new
        self.gnorm     = gnorm_new
        self.b         = b_new
        self.gnorm_min = min(self.gnorm_min, self.gnorm)


# =============================================================================
# Public interface
# =============================================================================

def minimize(grad, x0,
             tol=1e-6,
             max_iter=10_000,
             stopping_criterion=None,
             verbose=True,
             **solver_kwargs):
    """
    Minimize a smooth function using PolyTR.

    Only a gradient oracle is required — no function evaluations.

    Parameters
    ----------
    grad : callable
        Gradient oracle.  Called as ``grad(x)`` where x is a 1-D numpy array;
        must return a 1-D numpy array of the same length.
    x0 : array-like
        Starting point.
    tol : float
        Stop when ``||grad f(x)|| < tol``.  Ignored when
        *stopping_criterion* is provided.
    max_iter : int
        Hard iteration budget.
    stopping_criterion : callable, optional
        Custom stopping rule.  Called as ``stopping_criterion(solver)`` after
        each step.  The solver object exposes:
          - solver.x         — current iterate
          - solver.g         — current gradient
          - solver.gnorm     — ||g||
          - solver.gnorm_min — smallest ||g|| seen so far
        Return True to stop.
    verbose : bool
        Print a progress line every 100 iterations.
    **solver_kwargs
        Forwarded to PolyTR.  Useful knobs:
          - theta (float, default 1.1)  — polynomial decay exponent
          - eta   (float, default 0.85)  — trust-region radius scale
          - bmin  (float, default 1e-4) — minimum scaling factor

    Returns
    -------
    dict
        x        — solution found
        gnorm    — gradient norm at solution
        n_iter   — number of iterations taken
        success  — True if stopping criterion was met before max_iter
    """
    x0 = np.asarray(x0, dtype=float)
    opt = _PolyTR(grad, x0, **solver_kwargs)

    if stopping_criterion is None:
        def stopping_criterion(s):
            return s.gnorm < tol

    if verbose:
        print(f"{'Iter':>6}  {'||grad||':>12}")
        print("-" * 22)

    for k in range(1, max_iter + 1):
        opt.step()

        if verbose and (k == 1 or k % 100 == 0):
            print(f"{k:>6}  {opt.gnorm:>12.4e}")

        if not np.isfinite(opt.gnorm):
            break

        if stopping_criterion(opt):
            if verbose:
                print(f"{k:>6}  {opt.gnorm:>12.4e}  <- converged")
            return {"x": opt.x.copy(), "gnorm": opt.gnorm,
                    "n_iter": k, "success": True}

    return {"x": opt.x.copy(), "gnorm": opt.gnorm,
            "n_iter": k, "success": False}


# =============================================================================
# Example: Rosenbrock
# =============================================================================

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
        tol=1e-6,
        max_iter=10_000,
        
    )

    print()
    print(f"Converged : {result['success']}")
    print(f"Iterations: {result['n_iter']}")
    print(f"||grad||  : {result['gnorm']:.4e}")
    print(f"x*        : {np.round(result['x'], 6)}")
