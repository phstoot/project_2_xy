import simul as mod
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    test = mod.MonteCarlo_XY(100, 1)
    print(test.spins)
    print(np.rad2deg(test.spins))
    test.static_plot()