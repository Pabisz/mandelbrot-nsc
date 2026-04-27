"""
Arithmetic sensitivity of Mandelbrot trajectories

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def trajectory_divergence(N: int, MAX_ITER: int, TAU: float, x: np.ndarray, y: np.ndarray)-> np.ndarray:
    """
    Estimate divergence iteration differences between single and double precision trajectories.

    Parameters
    ----------
    N : int
        Grid size along each dimension (assumes NxN parameter grid).
    MAX_ITER : int
        Maximum number of iterations for the escape/divergence process.
    TAU : float
        Threshold for detecting divergence between float32 and float64 trajectories.
    x : np.ndarray
        Real-axis parameter values (shape typically (N,)).
    y : np.ndarray
        Imaginary-axis parameter values (shape typically (N,)).

    Returns
    -------
    np.ndarray
        Integer array of shape (N, N) where each entry represents the iteration
        index at which float32 and float64 trajectories diverged.
        If no divergence occurs, the value is MAX_ITER.
    """
    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)
    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)

    diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
    active = np.ones((N, N), dtype=bool)
    for k in range(MAX_ITER):
        if not active.any():
            break
        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]

        diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))

        newly = active & (diff > TAU)
        diverge[newly] = k
        active[newly] = False

    return diverge

def condition_number(N: int, MAX_ITER: int, x: np.ndarray, y: np.ndarray)-> np.ndarray:
    """
    Estimate numerical sensitivity (condition number) of escape-time iterations
    under perturbations in complex parameter space.

    Parameters
    ----------
    N : int
        Grid size along each dimension (assumes NxN parameter grid).
    MAX_ITER : int
        Maximum number of iterations for escape-time computation.
    x : np.ndarray
        Real-axis parameter values (shape typically (N,)).
    y : np.ndarray
        Imaginary-axis parameter values (shape typically (N,)).

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing estimated condition numbers.
        Values represent sensitivity of escape time to perturbations;
        entries may be NaN where escape time is zero.
    """
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)

    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)

    def escape_count(C: np.ndarray, max_iter: int) -> np.ndarray:
        z = np.zeros_like(C)
        cnt = np.full(C.shape, max_iter, dtype=np.int32)
        esc = np.zeros(C.shape, dtype=bool)

        for k in range(max_iter):
            z[~esc] = z[~esc]**2 + C[~esc]
            newly = ~esc & (np.abs(z) > 2.0)
            cnt[newly] = k
            esc[newly] = True

        return cnt

    n_base = escape_count(C, MAX_ITER).astype(float)
    n_perturb = escape_count(C + delta, MAX_ITER).astype(float)

    dn = np.abs(n_base - n_perturb)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

    return kappa

if __name__ == "__main__":
    N, MAX_ITER, TAU = 512, 1000, 0.01
    x = np.linspace(-0.7530, -0.7490, N)
    y = np.linspace( 0.0990, 0.1030, N)
    
    #Trajectory divergence
    diverge = trajectory_divergence(N, MAX_ITER, TAU, x, y)
    
    plt.figure()
    plt.imshow(diverge, cmap="plasma", origin="lower", extent=[-0.7530, -0.7490, 0.0990, 0.1030],)
    plt.colorbar(label="First divergence iteration")
    plt.title("Trajectory divergence")
    plt.show()

    #Condition number approximation
    kappa = condition_number(N, MAX_ITER, x, y)

    cmap_k = plt.cm.hot.copy()
    cmap_k.set_bad("0.25")

    vmax = np.nanpercentile(kappa, 99)

    plt.figure()
    plt.imshow(kappa, cmap=cmap_k, origin="lower", extent=[-0.7530, -0.7490, 0.0990, 0.1030], norm=LogNorm(vmin=1, vmax=vmax),)
    plt.colorbar(label=r"$\kappa(c)$ (log scale, $\kappa \geq 1$)")
    plt.title(r"Condition number approximation")
    plt.show()