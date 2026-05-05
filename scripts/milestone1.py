import numpy as np
import matplotlib.pyplot as plt
import json
import src.simul as simul
import src.utils as utils
import src.analysis as analysis

#
# !! start from project directory since /data path is hardcoded in here!!
#

def warmup():
    # for numba
    dummy_spins = np.zeros((4, 4))
    utils._run_sweeps(dummy_spins, 4, 1.0,
                np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64),
                np.zeros(16), np.zeros(16), n_sweeps=1)
    analysis.energy(dummy_spins, 50)

def main():
    utils.section('XY model simulation suite')
    # prepare parameters
    with open(f"data/high_res/batch_params.json") as f:
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
            # hot initial conditions
            sim = simul.MonteCarlo_XY(N, temps[i], start=f'{ic}')
            print(f'\nStarting sim: N = {N}, T = {temps[i]}, {ic} start, length = {sweeps[i]} sweeps...')
            sim.run(sweeps=sweeps[i], store=False, interval=sample_interval[i])
            # np.save(f'data/high_res/spins_{N}_T_{temps[i]}_hot.npy', np.array(hot.spins_hist))
            np.save(f'data/high_res/magn_{N}_T_{temps[i]}_hot.npy', np.array(sim.magn_hist))
    print('Done, bye')


if __name__ == '__main__':
    main()