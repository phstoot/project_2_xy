import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import njit
import pandas as pd


def section(title: str):
    """print to console in nice format"""
    width = 60
    print("\n" + "=" * width)
    print(f"{title.upper():^{width}}")
    print("=" * width + "\n")


def spacer(n: int = 2):
    """print to console in nice format"""
    print("\n" * n, end="")

def combine_measurements(group):
    """Used in one of the pipeline scripts for magnetic susceptibility, where we want to compile both the statistical error from """
    means = group['mean'].values
    errs = group['err'].values
    n = len(means)
    
    # combined mean
    grand_mean = np.mean(means)
    
    # statistical: propagated error on the mean
    stat_err = np.sqrt(np.sum(errs**2)) / n
    
    # systematic: sample std of the 6 means
    syst_err = np.std(means, ddof=1) / np.sqrt(n)
    
    # combined in quadrature
    total_err = np.sqrt(stat_err**2 + syst_err**2)
    
    return pd.Series({'mean': grand_mean, 'err': total_err})

# def hamiltonian(j_coupling= 1, spin_i, spin_j, magnetic= False):
#     if magnetic == True:
#         raise NotImplementedError('Not implemented')
    
#     H = -j_coupling #* np.sum()

def find_project_root():
    """Find project root from robust location (utils)
    """
    return Path(__file__).resolve().parents[1]
    

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
                - np.cos(new_theta - spins[xm, y]) + np.cos(initial_theta - spins[xm, y])
                - np.cos(new_theta - spins[xp, y]) + np.cos(initial_theta - spins[xp, y])
                - np.cos(new_theta - spins[x, ym]) + np.cos(initial_theta - spins[x, ym])
                - np.cos(new_theta - spins[x, yp]) + np.cos(initial_theta - spins[x, yp])
            )

            exponent = -beta * energy_diff
            if exponent > 0:
                exponent = 0.0

            if accepts[i] < np.exp(exponent):
                spins[x, y] = new_theta



def high_res_setup(batch: int = 0) -> Path:
    """Prepare the directory setup and parameter files of the high resolution simulation run.

    This run is meant to estimate correlation times. High resolution is therefore needed, 
    and long enough runs for good burn in and adequate precision of the autocorrelation function.
    We don't need a certain number of blocks to sample though,
    so the length of the runs has been determined through quantitative testing of X(t)

    Returns: data_dir (PosixPath) to store simulation output data
    """
    # prepare directories
    root = find_project_root()
    data_dir = root / f'data/high_res_{batch}'
    data_dir.mkdir(parents=True, exist_ok=True)

    # prepare params
    temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    sweeps = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 10000, 10000, 5000, 5000, 4000, 4000]
    discard = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 500, 500, 500, 500, 500]
    sample_interval = np.full(14, 1, dtype=int)
    N = 50
    ics = ['hot', 'cold']

    params = {
        "description": "XY simulation of size 50. High res run to estimate correlation time tau. Sweeps and burn-in time manually determined, sample interval = 1, hot+cold",
        "temps": temps,
        "sweeps": sweeps,
        "sample_interval": sample_interval.tolist(),
        "burn_in": discard,
        "N": N,
        "ics": ics
    }

    with open(data_dir/"batch_params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    return data_dir



def low_res_setup(batch:int=0) -> Path:
    """Prepare the directory setup and parameter files of the low resolution simulation run.

    This run is meant to calculate observables. Therefore a certain number of 'blocks' are needed
    for independent sampling, each block 16*correlation time long. Within a block we sample every ~0.25*tau,
    and we use 20 blocks + 2 blocks burn in to generate a large enough sample but don't push computing time. 
    It is up to the user to run multiple simulation batches to increase the data volume if that is needed
    for statistics.

    This setup can ONLY be run if the high_res simulation is already completed, and the correlation times
    can be loaded, which are needed for determining block size and thus number of sweeps.

    Returns: data_dir (PosixPath) to store simulation output data
    """
    # prepare directories
    root = find_project_root()
    data_dir = root / f'data/low_res_{batch}'
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / 'results'
    

    temps = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    try:
        taus = pd.read_csv(results_dir/'tau_temp.txt', delimiter='\t')[['temp','mean']]
    except:
        raise RuntimeError("Correlation times could not be loaded from 'results/tau_temp.txt'. Perhaps you forgot to run the high resolution scripts first?")
    taus_lowres = taus[taus['temp'].isin(temps)]['mean'].astype('int').reset_index(drop=True)
    sweeps = 22 * 16 * taus_lowres
    sample_interval = 0.25 * taus_lowres
    sample_interval = sample_interval.astype('int').replace(0, 1)
    block_size = np.array(((16 * taus_lowres) / sample_interval), dtype=int) # important conversion
    discard = 2 * block_size
    N = 50
    ics = ['hot', 'cold']

    params = {
        "description": "XY simulation of size 50. Low res run, tau extracted from high_res analysis. blocksize = 16*tau, sweeps = 22 blocks, burnin time = 2 blocks. sample interval = 0.25*tau, hot+cold",
        "temps": temps,
        "taus": taus_lowres.tolist(),  # convert from pandas to plain list
        "block_size": block_size.tolist(),
        "sweeps": sweeps.tolist(),
        "sample_interval": sample_interval.tolist(),
        "burn_in": discard.tolist(),
        "N": N,
        "ics": ics
    }

    with open(data_dir/"batch_params.json", "w") as f:
        json.dump(params, f, indent=2)

    return data_dir