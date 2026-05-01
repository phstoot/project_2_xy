import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul


def main():
    temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
    steps = 10**6 # watch out, may take a long time
    sample_interval = 1000
    N = [20, 100]

    for n in N:
        for T in temps:
            # hot initial conditions
            hot = simul.MonteCarlo_XY(n, T, start='hot')
            
            print(f'\nStarting sim: N = {n}, T = {T}, hot start')
            hot.run(steps=steps, store=True, interval=sample_interval)
            
            hot.static_image(save=True, fname=f'results/img_{n}_hotstart_T_{T}.pdf')
            
            xaxis = np.linspace(0, steps, steps // sample_interval)
            plt.plot(xaxis, hot.magn_hist)
            plt.xlabel('t (steps)')
            plt.ylabel(r'm ($M/N^2$)')
            plt.xlim(0,steps)
            plt.ylim(0, 1.1)
            plt.title(f'Magnetization of XY model\n $T$ = {T}, $J$ = 1, $k_B$ = 1, hot start')
            plt.savefig(f'results/magn_{n}_hotstart_T_{T}.pdf')
            plt.close('all')


            # cold initial conditions
            cold = simul.MonteCarlo_XY(n, T, start='cold')

            print(f'\nStarting sim: N = {n}, T = {T}, cold start') 
            cold.run(steps=steps, store=True, interval=sample_interval)
            
            cold.static_image(save=True, fname=f'results/img_{n}_coldstart_T_{T}.pdf')
            
            xaxis = np.linspace(0, steps, steps // sample_interval)
            plt.plot(xaxis, cold.magn_hist)
            plt.xlabel('t (steps)')
            plt.ylabel(r'm ($M/N^2$)')
            plt.xlim(0,steps)
            plt.ylim(0, 1.1)
            plt.title(f'Magnetization of XY model\n $T$ = {T}, $J$ = 1, $k_B$ = 1, cold start')
            plt.savefig(f'results/magn_{n}_coldstart_T_{T}.pdf')
            plt.close('all')

    print('done')


if __name__ == '__main__':
    main()