import numpy as np
import math
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
        """Initializes the simulation with given parameters.

        Parameters
        ----------
        [parameters here]
        """
        
        
        
        
        self.length_xy = length_xy
        self.temperature = temperature
        self.spins = self._init_spins()
        self._status = 'initialized'
        
        
        self.spins_hist: list = []
        self.saved_index_hist: list = [] # IF NOT USED, REMOVE LATER
        self.stepcount: int = 0
        

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
            scale=self.length_xy / 2,             # Inverse scale factor (higher = shorter arrows)
            width=0.15 / self.length_xy,           # Arrow shaft width
            # headwidth=3,           # Head width as multiple of shaft width
            # headlength=4,          # Head length as multiple of shaft width
            # headaxislength=3.5,    # Head length at shaft intersection
        )
        plt.colorbar(q, label='Angle (radians)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        
    def _propose_changed_index(self):
        """Private method: Propose index of the spin that will be changed for the new state. Every index occurs with same probability."""
        return np.random.randint(0,self.length_xy- 1), np.random.randint(0,self.length_xy- 1)
    
    def _propose_changed_theta(self, ind_x: int, ind_y):
        """Private method: Propose new theta for the inserted index."""
        return self.spins[ind_x, ind_y] + np.random.uniform(-np.pi, np.pi) + np.pi % (2*np.pi) - np.pi
    
    def acceptance_prob(self, energy_diff: float):
        """Calculates acceptance probability as a function of energy difference between proposed and initial state. J = beta = 1"""
        # print(min(1.0, math.exp(-energy_diff)))
        return min(1.0, math.exp(-energy_diff))

    def _step(self, alg: str = "verlet"):        
        """Private method: Propose and accept new state over Metropolis algorithm.
        Minimum image convention is applied."""
        changed_spin = self._propose_changed_index()
        new_theta = self._propose_changed_theta(ind_x = changed_spin[0], ind_y = changed_spin[1])
        
        energy_diff = 0
        index_shift = (-1,1)
        for dx in index_shift:
            for dy in index_shift:
                initial_theta = self.spins[changed_spin[0], changed_spin[1]]
                neighbour_theta = self.spins[np.mod(changed_spin[0] + dx,self.length_xy), np.mod(changed_spin[1] + dy,self.length_xy)]
                energy_diff += np.cos(neighbour_theta- new_theta)
                energy_diff -= np.cos(neighbour_theta - initial_theta)
                
        # Acceptance stage:
        P = self.acceptance_prob(energy_diff=energy_diff)
        if np.random.rand() < P:
            self.spins[changed_spin[0], changed_spin[1]] = new_theta        
        
    def _run(self, steps: int = 1000):
        """Private method for running the simulation without storing history and without status checks,
        used in equilibrate().
        """
        for step in range(steps):
            self._step()
            
    def equilibrate(
        self, steps_between = 1000
    ):
        """(Obviously needs some actual algorithm)"""
        
        if self._status == "equilibrated":
            raise RuntimeError(
                "System is already in equilibrium. Call run() to run simulation."
            )
        if self._status == "completed":
            raise RuntimeError("run() already called. Call reset() to start fresh.")
    
        self._run(steps=steps_between)
        
        

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

