######################################
# simple test script, free to change #
######################################
import os
import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul
import src.utils as utils
from pathlib import Path

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]
    print(utils.find_project_root())
    
    temps = np.arange(0.5, 2.5, 0.2)
    sweeps = 2000 # 5 million markov steps
    print(f'\nStarting sim..')
    sim = simul.MonteCarlo_XY(200, 1.5, start='cold')
    sim.run_live(sweeps=1000, batch_interval=20, save=False, fname='cold_to_hot.gif')
    sim.run(sweeps=sweeps, store=True, interval=20)
    # np.save('results/testmag.npy', np.array(sim.magn_hist))
    print('done')
    sim.static_image()
    plt.plot(sim.magn_hist)
    plt.xlabel('t (steps)')
    plt.ylabel('m (M/L^2)')
    # plt.xlim(0,steps)
    plt.title('Magnetization of XY model')
    # plt.savefig(f'results/magn.pdf')
    plt.close('all')

