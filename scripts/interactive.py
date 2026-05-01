import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

test = simul.MonteCarlo_XY(50, 0.5, start='hot')
coldtest = simul.MonteCarlo_XY(50, 1, start='cold')  

sweeps = 1000 # total of 2 500 000 steps

test.static_image()
coldtest.static_plot()

test.run(sweeps=sweeps, store=True, interval=10)

test.static_image()

xaxis = np.linspace(0, sweeps, sweeps // 10)
plt.plot(xaxis, test.magn_hist)
plt.xlabel('t (lattice sweeps)')
plt.ylabel(r'm ($M/N^2$)')
plt.xlim(0,sweeps)
plt.ylim(0, 1.1)
plt.title(f'Magnetization of XY model\n $T$ = 0.5, $J$ = 1, $k_B$ = 1, hot start')
# plt.savefig(f'results/magn_{n}_coldstart_T_{T}.pdf')
plt.show()
plt.close('all')



overflowtemps = np.array([0.1, 0.07, 0.06, 0.05])

beta = 1 / overflowtemps

dE = 8

def testflow(e):
    return np.exp(e)

exponents = np.linspace(0, 800, 61)
