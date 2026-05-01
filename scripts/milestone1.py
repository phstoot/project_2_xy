import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul


def main():
    temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    sweeps = 2000 # 5 million markov steps
    sample_interval = 1
    N = [50]

    for n in N:
        for T in temps:
            # hot initial conditions
            hot = simul.MonteCarlo_XY(n, T, start='hot')
            
            print(f'\nStarting sim: N = {n}, T = {T}, hot start')
            hot.run(sweeps=sweeps, store=True, interval=sample_interval)
            
            hot.static_image(show=False, save=True, fname=f'results/img_{n}_hotstart_T_{T}.pdf')
            
            xaxis = np.linspace(0, sweeps, sweeps // sample_interval)
            plt.plot(xaxis, hot.magn_hist)
            plt.xlabel('t (sweeps)')
            plt.ylabel(r'm ($M/N^2$)')
            plt.xlim(0,sweeps)
            plt.ylim(0, 1.1)
            plt.title(f'Magnetization of XY model\n N = {n}, $T$ = {T}, $J$ = 1, $k_B$ = 1, hot start')
            plt.savefig(f'results/magn_{n}_hotstart_T_{T}.pdf')
            plt.close('all')

            # cold initial conditions
            cold = simul.MonteCarlo_XY(n, T, start='cold')

            print(f'\nStarting sim: N = {n}, T = {T}, cold start') 
            cold.run(sweeps=sweeps, store=True, interval=sample_interval)
            
            cold.static_image(show=False, save=True, fname=f'results/img_{n}_coldstart_T_{T}.pdf')
            
            xaxis = np.linspace(0, sweeps, sweeps // sample_interval)
            plt.plot(xaxis, cold.magn_hist)
            plt.xlabel('t (sweeps)')
            plt.ylabel(r'm ($M/N^2$)')
            plt.xlim(0,sweeps)
            plt.ylim(0, 1.1)
            plt.title(f'Magnetization of XY model\n N = {n}, $T$ = {T}, $J$ = 1, $k_B$ = 1, cold start')
            plt.savefig(f'results/magn_{n}_coldstart_T_{T}.pdf')
            plt.close('all')

    print('done')


def save_magnetization_only():
    # for autocorrelation function
    temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    sweeps = 2000 # 5 million markov steps
    sample_interval = 1 # for precise correlation time estimation
    N = [50]

    for n in N:
        for T in temps:
            # hot initial conditions
            hot = simul.MonteCarlo_XY(n, T, start='hot')
            print(f'\nStarting sim: N = {n}, T = {T}, hot start')
            hot.run(sweeps=sweeps, store=True, interval=sample_interval)
            np.save(f'data/magn_{n}_hotstart_T_{T}.npy', hot.magn_hist)

            # cold initial conditions
            cold = simul.MonteCarlo_XY(n, T, start='cold')
            print(f'\nStarting sim: N = {n}, T = {T}, cold start') 
            cold.run(sweeps=sweeps, store=True, interval=sample_interval)
            np.save(f'data/magn_{n}_coldstart_T_{T}.npy', cold.magn_hist) 

    print('done') 

def save_spinstates():
    # for future calculations so we already have it. Later we can implement the infrastructure in the class
    temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    sweeps = 2000 # 5 million markov steps
    sample_interval = 1 # for precise measurements
    N = [50]

    for n in N:
        for T in temps:
            # hot initial conditions
            hot = simul.MonteCarlo_XY(n, T, start='hot')
            print(f'\nStarting sim: N = {n}, T = {T}, hot start')
            hot.run(sweeps=sweeps, store=True, interval=sample_interval)
            np.save(f'data/spins_{n}_hotstart_T_{T}.npy', np.array(hot.spins_hist))

            # cold initial conditions
            cold = simul.MonteCarlo_XY(n, T, start='cold')
            print(f'\nStarting sim: N = {n}, T = {T}, cold start') 
            cold.run(sweeps=sweeps, store=True, interval=sample_interval)
            np.save(f'data/spins_{n}_coldstart_T_{T}.npy', np.array(cold.spins_hist))


if __name__ == '__main__':
    # main()
    # save_magnetization_only()
    save_spinstates()