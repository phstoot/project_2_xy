import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

if __name__ == '__main__':
    # temps = np.arange(0.5, 2.5, 0.2)
    steps = 10**6 # watch out, may take a long time
    temps = [1,2]
    for T in temps:
        print(f'\nStarting sim for T = {T:.2f}...\n')
        sim = simul.MonteCarlo_XY(100, T, start='hot') # type: ignore
        sim.run(steps=steps, store=True)
        print('done')
        # sim.static_image()
        # plt.plot(sim.magn_hist)
        # plt.xlabel('t (steps)')
        # plt.ylabel('m (M/L^2)')
        # # plt.xlim(0,steps)
        # plt.title('Magnetization of XY model')
        # plt.savefig(f'results/magn_{T}.pdf')
        # plt.close('all')
