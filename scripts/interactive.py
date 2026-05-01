import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

test = simul.MonteCarlo_XY(10, 1, start='hot')
hottest = simul.MonteCarlo_XY(100, 10, start='hot') 
coldtest = simul.MonteCarlo_XY(100, 10, start='hot')  
small = simul.MonteCarlo_XY(20, 0.5, start='hot') 
big = simul.MonteCarlo_XY(100, 0.5, start='hot') 

    
steps = 10**6

test.static_plot()
hottest.static_plot()
coldtest.static_plot()

small.run(steps=steps, store=True, interval=1000)

small.static_image(save=True)



plt.plot(np.linspace(0, len(coldtest.magn_hist)*1000, len(coldtest.magn_hist)), coldtest.magn_hist)
plt.xlabel('t (steps)')
plt.ylabel(r'm (M/$N^2$)')
plt.xlim(0, len(coldtest.magn_hist)*1000)
plt.ylim(0,1)
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
