###############################################
# MILESTONE 2: calculate observables of system:
# This script runs several LOW-resolution 
# simulations and stores the data.
###############################################
import os
import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul
import src.utils as utils
import src.analysis as analysis
import pandas as pd
import json

def warmup():
    # for numba
    dummy_spins = np.zeros((4, 4))
    utils._run_sweeps(dummy_spins, 4, 1.0,
                np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64),
                np.zeros(16), np.zeros(16), n_sweeps=1)

def main(batch):
    # setup
    data_dir = utils.low_res_setup(batch)
    
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
            sim = simul.MonteCarlo_XY(N, temps[i], start=f'{ic}')
            print(f'\nStarting sim: N = {N}, T = {temps[i]}, {ic} start, length = {sweeps[i]} sweeps...')
            sim.run(sweeps=sweeps[i], store=True, interval=sample_interval[i])
            # np.save(data_dir/f'spins_{N}_T_{temps[i]}_hot.npy', np.array(hot.spins_hist)) # be careful, takes a lot of storage
            np.save(data_dir/f'energy_{N}_T_{temps[i]}_{ic}.npy', np.array(sim.e_hist))
            np.save(data_dir/f'magn_{N}_T_{temps[i]}_{ic}.npy', np.array(sim.magn_hist))
    print('Done, bye')

if __name__ == '__main__':
    for batch in range(3): # as many batches as is needed
        utils.section(f"Batch {batch}/2")
        main(batch)