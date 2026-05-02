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


# def hamiltonian(j_coupling= 1, spin_i, spin_j, magnetic= False):
#     if magnetic == True:
#         raise NotImplementedError('Not implemented')
    
#     H = -j_coupling #* np.sum()



@njit
def _run_sweeps(spins, length_xy, beta, xs, ys, deltas, accepts, n_sweeps):
    """Compiled core loop: runs all sweeps, modifies spins in place."""
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

            if accepts[i] < np.exp(min(0.0, -beta * energy_diff)):
                spins[x, y] = new_theta