######################################
# simple test script, free to change #
######################################
import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

if __name__ == '__main__':
    # temps = np.arange(0.5, 2.5, 0.2)
    sweeps = 2000 # 5 million markov steps
    print(f'\nStarting sim..')
    sim = simul.MonteCarlo_XY(50, 1, start='hot')
    sim.run(sweeps=sweeps, store=True, interval=20)
    np.save('results/testmag.npy', np.array(sim.magn_hist))
    print('done')
    # sim.static_image()
    # plt.plot(sim.magn_hist)
    # plt.xlabel('t (steps)')
    # plt.ylabel('m (M/L^2)')
    # # plt.xlim(0,steps)
    # plt.title('Magnetization of XY model')
    # plt.savefig(f'results/magn_{T}.pdf')
    # plt.close('all')
