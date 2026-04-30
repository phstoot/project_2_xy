import numpy as np
import matplotlib.pyplot as plt
import src.simul as simul

test = simul.MonteCarlo_XY(100, 1, start='hot')
hottest = simul.MonteCarlo_XY(100, 10, start='hot') 
coldtest = simul.MonteCarlo_XY(100, 0.01, start='hot') 

    
steps = 10**5 # watch out, scales quickly to long times

test._run(steps=steps)
hottest._run(steps=steps)
coldtest._run(steps=steps)
test.static_plot()
hottest.static_plot()
coldtest.static_plot()

coldtest.run(steps=steps, store=True)

coldtest.static_image()