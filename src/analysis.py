import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('project_2_xy/src')
# import simul
# import utils
import src.simul as simul
import src.utils as utils
from numba import njit


def autocorrelation(m, discard):
    # slice off equilibration time, for now let's do 500 timesteps
    m = m[discard:]
    t_max = len(m)

    X = np.zeros(len(m))
    # we are gonna iterate over all t's within the function
    time = np.arange(len(m))
    for t in range(len(m)):
        factor = 1 / (t_max - t)
        X[t] = factor * np.sum( m[0:(t_max-t)] * m[t:t_max] ) - factor * np.sum( m[0:(t_max-t)] ) * factor * np.sum( m[t:t_max] )

    return X

def correlation_time(x):
    stop_idx = np.where(x <= 0)[0][0]
    tau = np.sum( x[0:stop_idx] / x[0])
    return tau

def autocorrelation_curve(t, x_0, tau):
    x_t = x_0 * np.exp(-t / tau)
    return x_t

@njit
def energy(spins, length_xy):
    """Calculate total energy of a square grid of spins with a simple hamiltonian with coupling strength J=1 and no external field.
    Optimized for numba.

    Parameters
    ----------
    spins : array
        square array with spin states
    length_xy : int
        size of grid, number of spins in one direction

    Returns
    -------
    energy_per_spin : float
        the energy per spin of the system
    """
    energy = 0
    for x in range(length_xy):
        for y in range(length_xy):
            theta = spins[x, y]

            # avoid double counting so only right and up
            energy -= np.cos(theta - spins[x, (y+1) % length_xy]) # up neighbour
            energy -= np.cos(theta - spins[(x+1) % length_xy, y]) # right neighbour
    energy_per_spin = energy / length_xy**2
    return energy_per_spin

def independend_std(array, tau):
    """Calculate standard deviation of the mean for a correlated sample. Takes correlation time tau as input and corrects for 
    statistically dependent samples.
    """
    n_independent = len(array) / (2*tau)
    std_corrected = np.std(array) / np.sqrt(n_independent)
    return std_corrected

def magn_susc(magn_arr, temp, k_b, length_xy):
    """Calculate magnetic susceptibility from an array of sweeps. Important: use the non-absolute values of magnetization,
    stored as a 2D array [Mx, My] which is here needed as input.
    """
    # M2 = Mx**2 + My**2
    if not (magn_arr.ndim == 2 and magn_arr.shape[1] == 2):
        raise UserWarning("Magnetization array should have shape (n, 2)")
    else:
        return (1 / (temp * k_b * length_xy**2)) * (np.var(magn_arr[:,0]) + np.var(magn_arr[:,1]))

def specific_heat(energy_arr, temp, k_b, length_xy):
    return (1 / (temp**2 * k_b * length_xy**2)) * np.var(energy_arr)