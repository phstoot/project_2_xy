from matplotlib.widgets import EllipseSelector
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class MonteCarlo_XY:
    """_summary_
    """
    def __init__(
            self,
            length_xy : int = 50,
            temperature : float = 1,
            k_B : float = 1,
            start: str = 'cold' # either cold or hot
            ):
        """Initializes the simulation with given parameters.

        Parameters
        ----------
        [parameters here]
        """
        
        
        
        
        self.length_xy = length_xy
        self.temperature = temperature
        self.k_B = k_B
        self.spins = self._init_spins(start)
        self._status = 'initialized'
        self.start = start
        
        self.beta = 1 / (self.k_B * self.temperature)
        self.spins_hist: list = []
        self.magn_hist: list = []
        self.saved_index_hist: list = [] # IF NOT USED, REMOVE LATER
        self.stepcount: int = 0
        

    def __repr__(self) -> str:
        return (
            f"MonteCarlo_XY(length={self.length_xy}, temperature={self.temperature:.2f}, "
            f"num_particles={self.length_xy**2}, step= TO BE IMPLEMENTED, "
            f"status={self._status}"
            )

    def _init_spins(self, start):
        """Private method to initialize XY model with either random spins (start='hot') or aligned spins (start='cold')
        """
        if start == 'hot':
            spins = np.random.uniform(0, 2*np.pi, (self.length_xy, self.length_xy))
        elif start == 'cold':
            spins = np.full((self.length_xy, self.length_xy), (4/3)*np.pi)
        else:
            raise ValueError("choose 'hot' or 'cold' to start")
        return spins

    def static_plot(self):
        """Do a static visualisation of current system state using plt.quiver
        """
        #TODO now arrows at 0 are aligned on axis so only partly visible
        # create meshgrid for quiver
        X, Y = np.meshgrid(np.arange(self.length_xy), np.arange(self.length_xy)) #optional it says
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        
        fig, ax = plt.subplots(figsize=(8,8))
        q = ax.quiver(
            X, Y, U, V,
            self.spins,
            cmap='hsv', 
            clim=[0, 2*np.pi],
            pivot='mid',
            # scale_units='xy',      # Scale arrows relative to x,y axes
            scale= 0.9*self.length_xy,             # Inverse scale factor (higher = shorter arrows)
            width= 0.2 / self.length_xy,           # Arrow shaft width
            headwidth=2.8,           # Head width as multiple of shaft width
            headlength=6,          # Head length as multiple of shaft width
            headaxislength=5.6,    # Head length at shaft intersection
        )
        cbar = plt.colorbar(q, label='Angle (radians)', shrink=0.8, aspect=50)
        ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        labels = ['0', r'$\frac{1}{2}\pi$', r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$']
        cbar.set_ticks(ticks, labels=labels, fontsize=14)

        ax.set_aspect('equal')
        # ax.set_xlim(0, self.length_xy)
        # ax.set_ylim(0, self.length_xy)
        ax.margins(0)
        ax.set_xticks([])
        ax.set_yticks([]) 
        plt.tight_layout()
        plt.show()
    
    def static_image(self, show=True, save=False, fname='img.pdf'):
        """Do a static image of current system state with spins as pixels.
        """
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(
            self.spins,
            cmap='hsv',
            clim=[0, 2*np.pi], 
            interpolation='nearest' # no kernel
        )
        plt.title(f'XY model\n $T$ = {self.temperature}, $J$ = 1, $k_B$ = 1, {self.start} start')
        ax.set_xticks([])
        ax.set_yticks([]) 
        
        if save == True:
            plt.savefig(fname)
        if show == True:
            plt.show()
        plt.close('all')
            
    # TODO not sure if we need this function, but we probably do looking at milestone 2.
    # def equilibrate(
    #     self, steps_between = 1000
    # ):
    #     """(Obviously needs some actual algorithm)"""
        
    #     if self._status == "equilibrated":
    #         raise RuntimeError(
    #             "System is already in equilibrium. Call run() to run simulation."
    #         )
    #     if self._status == "completed":
    #         raise RuntimeError("run() already called. Call reset() to start fresh.")
    
    #     self._run(steps=steps_between)

    def _step(self, x, y, delta, accept):
        """Private optimized step function implementing Metropolis Hastings algorithm. 
        All random numbers for a run are generated in run method beforehand, and fed via _sweep method into _step method to avoid loop overhead. 
        Periodic boundary conditions are applied.

        Parameters
        ----------
        x : int
            the x index of the changed spin
        y : int
            the y index of the changed spin
        delta : float
            the spin change
        accept : float
            random draw for acceptance logic
        """
        # calculate energy difference
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        initial_theta = self.spins[x,y] 
        energy_diff = 0 # unnecessary to define, but technically it could become unbound in acceptance block, if for some reason loop doesn't work
        for dx, dy in neighbours: 
                nx = (x + dx) % self.length_xy
                ny = (y + dy) % self.length_xy
                neighbour_theta = self.spins[nx, ny]
                energy_diff += -np.cos(delta - neighbour_theta) + np.cos(initial_theta - neighbour_theta) # dE = final - initial
                
        # Acceptance stage:
        exponent = min(0, -self.beta * energy_diff) # prevent overflow error by choosing before evaluating exponent
        if accept < np.exp(exponent):
            self.spins[x,y] = delta

    def _sweep(self, sweep_counter, xs, ys, deltas, accepts):
        """Perform one full lattice sweep. One lattice sweep consists of N^2 _steps,
        giving each spin a chance to be flipped. This sweep is used as timestep in the calculation of statistics of interest. 
        The random numbers arrays generated in run are passed to _sweep, along with a counter indicating how many sweeps are done.
        _sweep iterates over random number arrays from sweep_count*N^2 to (sweep_count + 1) * N^2, and passes the individual random nrs on to _step.
        Every sweep call effectively iterates over a slice of the total random nr arrays.
        """
        for i in range((sweep_counter * self.length_xy**2), ((sweep_counter+1)* self.length_xy**2)): # take correct slice of random numbers
            self._step(xs[i], ys[i], deltas[i], accepts[i])

    def run(self, sweeps: int = 1000, store: bool=True, interval: int = 10):
        """Optimized run method to run the simulation for a number of lattice sweeps in a Monte Carlo Markov Chain. 
        Draws all needed random numbers at the start of the run, to prevent overhead during the steps.
        For more info on the structure concerning _sweep, and how random numbers are passed on in iterations, see docstrings _sweep and _step

        Parameters
        ----------
        sweeps : int, optional, default 1000
            the number of lattice sweeps / timesteps for the run. Each sweep performs N^2 Markov steps
        store : bool, optional, default True
            whether to store history arrays
        interval : int, optional, default 10
            sample to history arrays in interval, counted in sweeps
        """
        # Pre-generate all random numbers at once
        size = sweeps * self.length_xy**2 # total size of the run in Markov Chain steps
        rng = np.random.default_rng()  # modern API, apparently faster than np.random
        xs = rng.integers(0, self.length_xy, size=size)
        ys = rng.integers(0, self.length_xy, size=size)
        deltas = rng.uniform(0, 2 * np.pi, size=size)
        accepts = rng.random(size=size)  # for the acceptance draw

        sweep_counter: int = 0
        for i in tqdm(range(sweeps)):
            if store == True and i % interval == 0:
                self.magn_hist.append(self._calculate_magnetization())
                self.spins_hist.append(self.spins)
            self._sweep(sweep_counter, xs, ys, deltas, accepts)
            sweep_counter += 1

    def _total_energy(self):

        pass
    
    def _calculate_magnetization(self):
        """Calculate magnetization per spin m = M/N^2 where M = sum(spins).
        """
        M = np.abs(np.sum(np.exp(1j * self.spins)))
        m = M / self.length_xy**2
        return m
    
    def _calculate_autocorrelation(self, t):




        pass
    
    def calculate_specific_heat(self):
        return
    
    def calculate_susceptibility(self):
        return

