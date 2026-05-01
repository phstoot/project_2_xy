import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

test = simul.MonteCarlo_XY(10, 1, start='hot')
hottest = simul.MonteCarlo_XY(100, 10, start='hot') 
coldtest = simul.MonteCarlo_XY(100, 0.001, start='hot') 

    
steps = 10**6 # watch out, scales quickly to long times

test.static_plot()
hottest.static_plot()
coldtest.static_plot()

coldtest.run(steps=steps, store=True)

coldtest.static_image(save=False)


plt.plot(np.linspace(0, len(coldtest.magn_hist)*100, len(coldtest.magn_hist)+1), coldtest.magn_hist)
plt.xlabel('t (steps)')
plt.ylabel(r'm (M/$N^2$)')
plt.xlim(0, len(coldtest.magn_hist))
plt.ylim(0,0.5)
plt.title(f'Magnetization of XY model, T = {coldtest.temperature}')
# plt.savefig(f'results/magn_{T}.pdf')
plt.show()
# plt.close('all')


overflowtemps = np.array([0.1, 0.07, 0.06, 0.05])

beta = 1 / overflowtemps

dE = 8

def testflow(e):
    return np.exp(e)

exponents = np.linspace(0, 800, 61)
