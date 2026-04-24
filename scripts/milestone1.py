import numpy as np
import matplotlib.pyplot as plt
import tqdm

class MonteCarlo_XY:
    """_summary_
    """
    def __init__(
            self,
            length_xy : int = 4,
            temperature : float = 1,
            ):
        self.length_xy = length_xy
        self.temperature = temperature
        self.spins = self._init_spins()
        self._status = 'initialized'
        pass

    def __repr__(self) -> str:
        return (
            f"MonteCarlo_XY(length={self.length_xy}, temperature={self.temperature:.2f}, "
            f"num_particles={self.length_xy**2}, step= TO BE IMPLEMENTED, "
            )
        pass

    def _init_spins(self):
        spins = np.random.uniform(-np.pi, np.pi, (self.length_xy, self.length_xy))
        return spins

    def static_plot(self):
        """Do a static visualisation of current system state
        """

    def step(self): 
        pass
    

    def _total_energy(self):
        
        pass
    
    def _calculate_magnetization(self):
        return 
    
    def _calculate_autocorrelation(self):
    
    
    def calculate_specific_heat(self):
        return
    
    def calculate_susceptibility(self):
        return

