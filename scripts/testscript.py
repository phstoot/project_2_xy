import sys
sys.path.append('project_2_xy')
# import src.simul as mod


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import simul # src is added to path so direct import


if __name__ == '__main__':
    test = simul.MonteCarlo_XY(10, 1)
    print(test.spins)
    print(np.rad2deg(test.spins))
    test.static_plot()
    
    test.equilibrate(steps_between=100000)
    test.static_plot() # Issue: Colour map suggests that arrows are well aligned, arrows themselves still seem to point in different directions.
    
    