import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import njit


def section(title: str):
    """print to console in nice format"""
    width = 60
    print("\n" + "=" * width)
    print(f"{title.upper():^{width}}")
    print("=" * width + "\n")


def spacer(n: int = 2):
    """print to console in nice format"""
    print("\n" * n, end="")



@njit
def _run_sweeps(spins: np.ndarray, length_xy: int, beta: float, xs: np.ndarray, ys: np.ndarray, deltas: np.ndarray, accepts: np.ndarray, n_sweeps: int):
    """Compiled core loop: runs batches of sweeps by letting numba handle the computation.
    Implements periodic boundary conditions over modulo indexing.
    Energy differences only considers the the contributions changed by the proposed update. 
    To avoid evaluating large exponentials, the exponents are minimized first before being passed to np.exp, and if they are positive.
    
    Parameters
    ----------
    spins : array
        array with spin angles at each lattice site
    length_xy : int
        size of the grid
    beta : float
        inverse temperature
    xs : array
        x-index of the spins to be updated
    ys : array
        y-index of the spins to be updated
    deltas : array
        proposed changes to the spin angles
    accepts : array
        random numbers for acceptance criterion
    n_sweeps : int
        number of sweeps to run

    Returns
    -------
    None
    """
    steps_per_sweep = length_xy * length_xy
    for sweep in range(n_sweeps):
        for i in range(sweep * steps_per_sweep, (sweep + 1) * steps_per_sweep):
            x = xs[i]
            y = ys[i]
            initial_theta = spins[x, y]
            new_theta = deltas[i]

            xm = (x - 1) % length_xy
            xp = (x + 1) % length_xy
            ym = (y - 1) % length_xy
            yp = (y + 1) % length_xy

            energy_diff = (
                -np.cos(new_theta - spins[xm, y]) + np.cos(initial_theta - spins[xm, y])
                - np.cos(new_theta - spins[xp, y]) + np.cos(initial_theta - spins[xp, y])
                - np.cos(new_theta - spins[x, ym]) + np.cos(initial_theta - spins[x, ym])
                - np.cos(new_theta - spins[x, yp]) + np.cos(initial_theta - spins[x, yp])
            )

            exponent = -beta * energy_diff
            if exponent > 0:
                exponent = 0.0

            if accepts[i] < np.exp(exponent):
                spins[x, y] = new_theta