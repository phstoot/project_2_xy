import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

# for the final measurements, we need a lot of sweeps for the systems around the equilibrium.
# the critical system has a tau ~500, so probably needs 1000+ sweeps for equilibrium. We then need about 20 blocks of 16*tau 
# to get accurate measurements of the statistics, so that would mean ~160 000 sweeps, plus a few tau for equilibration.


def main():
    temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    sweeps = [162000, 162000, 162000, 162000, 16200, 4000, 4000, 2000, 2000, 2000, 2000]
    sample_interval = [80, 80, 80, 80, 8, 1, 1, 1, 1, 1, 1]
    N = 50

    for i in range(len(temps)):
        # hot initial conditions
        hot = simul.MonteCarlo_XY(N, temps[i], start='hot')
        print(f'\nStarting sim: N = {N}, T = {temps[i]}, hot start, length = {sweeps[i]} sweeps...')
        hot.run(sweeps=sweeps[i], store=True, interval=sample_interval[i])
        np.save(f'data/spins_{N}_hotstart_T_{temps[i]}.npy', np.array(hot.spins_hist))

        # cold initial conditions
        cold = simul.MonteCarlo_XY(N, temps[i], start='cold')
        print(f'\nStarting sim: N = {N}, T = {temps[i]}, cold start, length = {sweeps[i]} sweeps...') 
        cold.run(sweeps=sweeps[i], store=True, interval=sample_interval[i])
        np.save(f'data/spins_{N}_coldstart_T_{temps[i]}.npy', np.array(cold.spins_hist))
    print('Done, bye')


if __name__ == '__main__':
    main()