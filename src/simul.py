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

    def __repr__(self) -> str:
        return (
            f"MonteCarlo_XY(length={self.length_xy}, temperature={self.temperature:.2f}, "
            f"num_particles={self.length_xy**2}, step= TO BE IMPLEMENTED, "
            f"status={self._status}"
            )

    def _init_spins(self):
        spins = np.random.uniform(-np.pi, np.pi, (self.length_xy, self.length_xy))
        return spins

    def static_plot(self):
        """Do a static visualisation of current system state using plt.quiver
        """
        # create meshgrid for quiver
        X, Y = np.meshgrid(np.arange(self.length_xy), np.arange(self.length_xy)) #optional it says
        # angles = np.rad2deg(self.spins)

        U = np.cos(self.spins)
        V = np.sin(self.spins)
        fig, ax = plt.subplots(figsize=(10,10))
        
        q = ax.quiver(
            X, Y, U, V,
            self.spins,
            cmap='hsv', 
            clim=[-np.pi, np.pi],
            pivot='mid',
            # scale_units='xy',      # Scale arrows relative to x,y axes
            # scale=4,             # Inverse scale factor (higher = shorter arrows)
            # width=0.003,           # Arrow shaft width
            # headwidth=3,           # Head width as multiple of shaft width
            # headlength=4,          # Head length as multiple of shaft width
            # headaxislength=3.5,    # Head length at shaft intersection
        )
        plt.colorbar(q, label='Angle (radians)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def step(self): 
        pass
    

    def _total_energy(self):
        pass
    
    def _calculate_magnetization(self):
        return 
    
    def _calculate_autocorrelation(self):
        pass
    
    def calculate_specific_heat(self):
        return
    
    def calculate_susceptibility(self):
        return

