import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('project_2_xy/src')
# import simul 
# import utils
# import analysis
import src.simul as simul
import src.utils as utils
import src.analysis as analysis


def warmup():
    dummy_spins = np.zeros((4, 4))
    utils._run_sweeps(dummy_spins, 4, 1.0,
                np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64),
                np.zeros(16), np.zeros(16), n_sweeps=1)


warmup()
test = simul.MonteCarlo_XY(50, 0.9,start='hot', low_memory=False)
coldtest = simul.MonteCarlo_XY(50, 1, start='cold', low_memory=True)  

sweeps = 1000 # total of 2 500 000 steps

# test.static_image()
# coldtest.static_plot()

# test.run(sweeps=sweeps)

# coldtest.run(sweeps=sweeps)
# test.static_image()
# coldtest.static_image()



test.run_live(sweeps=sweeps, save=False, show=True)

test.static_image()

coldtest.run_live(sweeps=sweeps, save=False, show=True)
coldtest.static_image()

# xaxis = np.linspace(0, sweeps, sweeps // 10)
# plt.plot(xaxis, test.magn_hist)
# plt.xlabel('t (lattice sweeps)')
# plt.ylabel(r'm ($M/N^2$)')
# plt.xlim(0,sweeps)
# plt.ylim(0, 1.1)
# plt.title(f'Magnetization of XY model\n $T$ = 0.5, $J$ = 1, $k_B$ = 1, hot start')
# # plt.savefig(f'results/magn_{n}_coldstart_T_{T}.pdf')
# plt.show()
# plt.close('all')



overflowtemps = np.array([0.1, 0.07, 0.06, 0.05])

beta = 1 / overflowtemps

dE = 8

def testflow(e):
    return np.exp(e)

exponents= np.linspace(0, 800, 61)

e_hist = []
for state in test.spins_hist:
    e_hist.append(analysis.energy(state, 50))