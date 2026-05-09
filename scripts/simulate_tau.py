###############################################
# MILESTONE 1: estimate correlation time tau:
# This script runs several HIGH-resolution 
# simulations and stores the data.
###############################################
import os
import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul
import src.utils as utils
import src.analysis as analysis
import json

low_memory = True # If True, code is slightly slower but less memory-intensive.

def warmup():
    # for numba
    dummy_spins = np.zeros((4, 4))
    utils._run_sweeps(dummy_spins, 4, 1.0,
                np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64),
                np.zeros(16), np.zeros(16), n_sweeps=1)
    analysis.energy(dummy_spins, 50)

def main(batch: int = 0):
    # setup
    data_dir = utils.high_res_setup(batch)

    utils.section('XY model simulation suite')
    
    # prepare parameters
    with open(data_dir/"batch_params.json") as f:
        params = json.load(f)
    print("Parameters description:\n")
    print(params['description'])
    temps = params["temps"]
    sweeps = params["sweeps"]
    sample_interval = params['sample_interval']
    ics = params['ics']
    N = params['N']

    warmup()
    for ic in ics:
        for i in range(len(temps)):
            sim = simul.MonteCarlo_XY(N, temps[i], start=f'{ic}', low_memory=low_memory)
            print(f'\nStarting sim: N = {N}, T = {temps[i]}, {ic} start, length = {sweeps[i]} sweeps...')
            sim.run(sweeps=sweeps[i], store=True, interval=sample_interval[i], abs=False)
            # np.save(data_dir/f'spins_{N}_T_{temps[i]}_hot.npy', np.array(hot.spins_hist)) # be careful, takes a lot of storage
            np.save(data_dir/f'magn_{N}_T_{temps[i]}_{ic}.npy', np.array(sim.magn_hist)) 
            np.save(data_dir/f'energy_{N}_T_{temps[i]}_{ic}.npy', np.array(sim.e_hist)) 
            np.save(data_dir/f'v_dens_{N}_T_{temps[i]}_{ic}.npy', np.array(sim.v_dens_hist))
    print('Done, bye')

if __name__ == '__main__':
    for batch in range(4): # as many batches as is needed
        utils.section(f"Batch {batch}/3")
        main(batch)