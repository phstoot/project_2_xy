from matplotlib.widgets import EllipseSelector
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
# import sys
# sys.path.append('project_2_xy/src')
from src.utils import _run_sweeps, find_vortices
import src.analysis as analysis

class MonteCarlo_XY:
    """see __init__
    """
    def __init__(
            self,
            length_xy : int = 50,
            temperature : float = 1,
            k_B : float = 1,
            start: str = 'cold', # either cold or hot
            low_memory: bool = False,
            h: float = 0
            ):
        """
        Monte Carlo XY Model simulation
        -------------------------------
        This class implements a Monte Carlo simulation of the 2D XY model on a square lattice. It uses the 
        Metropolis Hastings algorithm to stochastically sample phase space for a system at a given temperature and create a 
        Markov chain of system states.
        Instances of the class can be run with or without animation, and can store the history of the spins,
        magnetization and energy per spin in accessible arrays for later usage. 

        The core of the simulation implements the numba JIT compiler to significantly speed up runtime. In a 
        standard user laptop/computer, this code should be able to do about 10^7 Monte Carlo steps per second. 
        This makes the class exceptionally suitable to obtain large datasets for further analysis of XY model 
        observables, like correlation time, magnetic susceptibility and specific heat. 

        Parameters
        ----------
        length_xy : int, optional
            length of grid, by default 50
        temperature : float, optional
            temperature bath of system, by default 1
        k_B : float, optional
            boltzmann constant, by default 1
        start : str, optional
            initial conditions, either 'hot' (random spins) or 'cold' (aligned spins), by default 'cold'
        low_memory : bool, optional
            if code is run memory-intensive or slower, by default false
        h : float, optional
            external magnetic field, by default 0

        Main methods
        ------------

        static_plot(): Do a static visualisation of current system state using plt.quiver

        static_image(): Do a static image of current system state with spins as pixels.

        run(): Runs the simulation for a number of lattice sweeps in a Monte Carlo Markov Chain.

        run_live(): Runs a live simulation + animation of the lattice evolution.
        """

        self.length_xy = length_xy
        self.temperature = temperature
        self.k_B = k_B
        self.spins = self._init_spins(start)
        self._status = 'initialized'
        self.start = start
        self.low_memory = low_memory
        self.h = h
        self.beta = 1 / (self.k_B * self.temperature)
        self.spins_hist: list = []
        self.magn_hist: list = []
        self.e_hist: list = []
        self.v_dens_hist: list = []
        self.saved_index_hist: list = [] # IF NOT USED, REMOVE LATER
        self.stepcount: int = 0
        

    def __repr__(self) -> str:
        return (
            f"MonteCarlo_XY(length={self.length_xy}, temperature={self.temperature:.2f}, "
            f"num_particles={self.length_xy**2}, saved states = {len(self.magn_hist)}"
            f"status={self._status}"
            )

    def _init_spins(self, start):
        """Private method to initialize XY model with either random spins (start='hot') or aligned spins (start='cold'), or some funky version.
        
        Parameters
        ----------
        start : str
            Initial conditions for the spins ('hot', 'cold', or 'funky')

        Returns
        -------
        spins : np.ndarray
            Initial spin configuration
        """
        spins = np.zeros((self.length_xy, self.length_xy))
        if start == 'hot':
            spins = np.random.uniform(0, 2*np.pi, (self.length_xy, self.length_xy))
        elif start == 'cold':
            spins = np.full((self.length_xy, self.length_xy), (4/3)*np.pi)
        elif start == 'funky':
            print('You funky bastard...')
            do = np.random.randint(0,3)
            if do == 0:
                print('Choosing two opposing halves (try static_image before running)')
                spins = np.zeros((self.length_xy, self.length_xy))
                mid = self.length_xy // 2
                spins[:, :mid] = 0
                spins[:, mid:] = np.pi
            elif do == 1:
                print('Choosing four quadrants (try static_image before running)')
                spins = np.zeros((self.length_xy, self.length_xy))
                mid = self.length_xy // 2
                spins[:mid, :mid] = 0
                spins[:mid, mid:] = np.pi / 2
                spins[mid:, :mid] = np.pi
                spins[mid:, mid:] = 3*np.pi / 2
            elif do == 2:
                print('Choosing center region (try static_image before running)')
                L = self.length_xy
                x, y = np.indices((L, L))
                cx = cy = L // 2
                R = L // 4
                w = 2.0   # interface width
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                # smooth transition function
                f = 0.5 * (1 - np.tanh((r - R)/w))
                # interpolate between two angles
                theta_inside = np.pi
                theta_outside = 0

                spins = theta_outside + f * (theta_inside - theta_outside)
                spins = np.mod(spins, 2*np.pi)
            elif do == 3:
                print('Choosing gradient (try static_image before running)')
                x, y = np.indices((self.length_xy, self.length_xy))
                spins = 2*np.pi * x / self.length_xy
            elif do == 4:
                print('Choosing vortex (try static image before running)')
                x, y = np.indices((self.length_xy, self.length_xy))
                cx = cy = self.length_xy // 2
                spins = np.arctan2(y - cy, x - cx)
                spins = np.mod(spins, 2*np.pi)
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
        
        vortices = find_vortices(self.spins)
        vx = []
        vy = []
        avx = []
        avy = []
        for x, y, q in vortices:
            if q == 1:
                vx.append(y)
                vy.append(x)
            elif q == -1:
                avx.append(y)
                avy.append(x)
        
        vortex_scatter = ax.scatter(vx, vy, color='darkgrey', marker='o', s=100, label='vortex')
        antivortex_scatter = ax.scatter(avx, avy, color='black', marker='o', s=100, label='antivortex')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def static_image(self, show=True, save=False, fname='img.pdf'):
        """Do a static image of current system state with spins as pixels.
        """
        fig, ax = plt.subplots(figsize=(8,8))
        image = ax.imshow(
            self.spins,
            cmap='hsv',
            clim=[0, 2*np.pi], 
            interpolation='nearest' # no kernel
        )
        
        vortices = find_vortices(self.spins)
        vx = []
        vy = []
        avx = []
        avy = []
        for x, y, q in vortices:
            if q == 1:
                vx.append(y)
                vy.append(x)
            elif q == -1:
                avx.append(y)
                avy.append(x)
        
        vortex_scatter = ax.scatter(vx, vy, color='darkgrey', marker='o', s=100, label='vortex')
        antivortex_scatter = ax.scatter(avx, avy, color='black', marker='o', s=100, label='antivortex')
        
        plt.title(f'XY model\n $T$ = {self.temperature}, $J$ = 1, $k_B$ = 1, {self.start} start')
        plt.legend()
        ax.set_xticks([])
        ax.set_yticks([]) 
        
        if save == True:
            plt.savefig(fname)
        if show == True:
            plt.show()
        plt.close('all')

    def _step(self, x:int, y:int, delta:float, accept:float):
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
                energy_diff += -np.cos(delta - neighbour_theta) - self.h * np.cos(delta) + np.cos(initial_theta - neighbour_theta)  + self.h * np.cos(initial_theta) # dE = final - initial
                
        # Acceptance stage:
        exponent = min(0, -self.beta * energy_diff) # prevent overflow error by choosing before evaluating exponent
        if accept < np.exp(exponent):
            self.spins[x,y] = delta

    def _sweep(self, sweep_counter: int, xs: np.ndarray, ys: np.ndarray, deltas: np.ndarray, accepts: np.ndarray):
        """Perform one full lattice sweep. One lattice sweep consists of N^2 _steps,
        giving each spin a chance to be flipped. This sweep is used as timestep in the calculation of statistics of interest. 
        The random numbers arrays generated in run are passed to _sweep, along with a counter indicating how many sweeps are done.
        _sweep iterates over random number arrays from sweep_count*N^2 to (sweep_count + 1) * N^2, and passes the individual random nrs on to _step.
        Every sweep call effectively iterates over a slice of the total random nr arrays.
        """
        for i in range((sweep_counter * self.length_xy**2), ((sweep_counter+1)* self.length_xy**2)): # take correct slice of random numbers
            self._step(xs[i], ys[i], deltas[i], accepts[i])
  
    def run(self, sweeps: int = 1000, store: bool = True, interval: int = 10, abs=False):
        """Runs the simulation for a number of lattice sweeps in a Monte Carlo Markov Chain.
        Optimized by generating all random numbers at the start of the run, to prevent overhead during the steps.
        Uses numba from an external method to efficiently run the sweeps, calculating evolution in batches.
        Used to obtain large datasets.
        Depending on the memory setting, random numbers are either wholly generated or for each batch separately.
        
        Parameters
        ---------- 
        sweeps : int, optional, default 1000
            the number of lattice sweeps / timesteps for the run. Each sweep performs N^2 Markov steps.
        store : bool, optional, default True
            whether to store history arrays
        interval : int, optional, default 10
            sample to history arrays in interval, counted in sweeps
        abs : bool, optional, default False
            whether to take absolute value when calculating the magnetization, by default False.
        """
        size = sweeps * self.length_xy**2
        rng = np.random.default_rng()
        
        if self.low_memory == False:
            xs = rng.integers(0, self.length_xy, size=size, dtype=np.int64)
            ys = rng.integers(0, self.length_xy, size=size, dtype=np.int64)
            deltas = rng.uniform(0, 2 * np.pi, size=size)
            accepts = rng.random(size=size)

        for i in tqdm(range(0, sweeps, interval), ascii="▏▎▍▌▋▊▉█", colour="#457b9d"): 
            if store:
                self.magn_hist.append(self._calculate_magnetization(abs=abs))
                self.e_hist.append(analysis.energy(self.spins, self.length_xy))
                self.spins_hist.append(self.spins.copy())
                self.v_dens_hist.append(self._calculate_vortex_density()[0])
            batch = min(interval, sweeps - i)
            
            if self.low_memory:
                xs = rng.integers(0, self.length_xy, size=batch * self.length_xy**2, dtype=np.int64)
                ys = rng.integers(0, self.length_xy, size=batch * self.length_xy**2, dtype=np.int64)
                deltas = rng.uniform(0, 2 * np.pi, size=batch * self.length_xy**2)
                accepts = rng.random(size=batch * self.length_xy**2)
                _run_sweeps(self.spins, self.length_xy, self.beta,
                            xs, ys, deltas, accepts,
                            n_sweeps=batch, field=self.h)
                
            else:
                start = i * self.length_xy**2
                end = (i + batch) * self.length_xy**2
                _run_sweeps(self.spins, self.length_xy, self.beta,
                            xs[start:end], ys[start:end], deltas[start:end], accepts[start:end], # type: ignore
                            n_sweeps=batch, field=self.h)
    
    def _update_animation(self,frame: int, store: bool = True):     
        """Update function for live animation. Called by FuncAnimation for every frame.
        Runs a batch of sweeps calling the numba optimized _run_sweeps method, and updates the image and title of the plot.
        
        Parameters
        ----------
        frame : int
            the current frame number, used to calculate which slice of random numbers to use for the batch
        store : bool, optional, default True
            whether to store history arrays
        
        Returns
        -------
        tuple
            the updated image and title objects for FuncAnimation
        """
        if store:
            self.magn_hist.append(self._calculate_magnetization())
            self.e_hist.append(self._calculate_energy())
            self.spins_hist.append(self.spins.copy())
            self.v_dens_hist.append(self._calculate_vortex_density()[0])

        if self.low_memory:
            xs = self.rng.integers(0, self.length_xy, size=self.batch_interval *self.length_xy**2, dtype=np.int64)
            ys = self.rng.integers(0, self.length_xy, size=self.batch_interval *self.length_xy**2, dtype=np.int64)
            deltas = self.rng.uniform(0, 2*np.pi, size=self.batch_interval *self.length_xy**2)
            accepts = self.rng.random(size=self.batch_interval *self.length_xy**2)
        else:
            start = frame * self.batch_interval * self.length_xy**2
            xs = self.xs[start:start+ self.batch_interval *self.length_xy**2]
            ys = self.ys[start:start+self.batch_interval *self.length_xy**2]
            deltas = self.deltas[start:start+self.batch_interval *self.length_xy**2]
            accepts = self.accepts[start:start+self.batch_interval *self.length_xy**2]

        _run_sweeps(self.spins, self.length_xy, self.beta,
                        xs, ys, deltas, accepts,
                        n_sweeps=self.batch_interval, field=self.h)
        
        self.image.set_array(self.spins)
        self.title.set_text(f'Sweep {self.batch_interval *(frame+1)} | T = {self.temperature} | {self.start} start')

        return (self.image, self.title)
    
    
    def run_live(self, sweeps: int = 1000, batch_interval: int = 10, anim_interval: int = 20, save: bool = False, show: bool = True, fname: str = 'animation.gif'):
        """Runs a live simulation of the lattice evolution. Calls _update_animation for every frame.
        Frames are calculated in batches of sweeps.
        If low_memory == False, all random numbers are generated at the start of the run. 
        If low_memory == True, random numbers are generated for each batch separately in _update_animation.
        
        Parameters
        ----------
        sweeps : int, optional, default 1000
            the number of sweeps to run
        batch_interval : int, optional, default 10
            the number of sweeps to run before updating the animation
        anim_interval : int, optional, default 20
            the interval between animation frames in milliseconds
        save : bool, optional, default False
            whether to save the animation as a GIF
        show : bool, optional, default True
            whether to display the animation
        fname : str, optional, default 'animation.gif'
            the filename to save the animation as

        Returns
        -------
        None
        """
        size = sweeps * self.length_xy**2
        self.batch_interval = batch_interval
        
        self.rng = np.random.default_rng()
        
        if self.low_memory == False:
            rng = np.random.default_rng()
            self.xs = self.rng.integers(0, self.length_xy, size=size, dtype=np.int64)
            self.ys = self.rng.integers(0, self.length_xy, size=size, dtype=np.int64)
            self.deltas = self.rng.uniform(0, 2*np.pi, size=size)
            self.accepts = self.rng.random(size=size)

        fig, ax = plt.subplots(figsize=(6, 6))
        self.image = ax.imshow(
            self.spins,
            cmap='hsv',
            vmin=0, vmax=2*np.pi,
            interpolation='nearest'
        )

        ax.set_xticks([])
        ax.set_yticks([])
        self.title = ax.set_title(f'Sweep 0 | T = {self.temperature} | {self.start} start')

        anim = animation.FuncAnimation(
            fig,
            self._update_animation,
            frames=int(np.floor(sweeps / self.batch_interval)),
            interval=anim_interval,
            blit=False,
            repeat=False
        )
        
        if save == True:
            anim.save(fname, writer='pillow')  # for GIF
        if show == True:
            plt.show()
        plt.close('all')

        # Delete unnecessary large random number arrays to free memory
        self.xs = []
        self.ys = []
        self.deltas = []
        self.accepts = []
        self._status = "completed"
    

    def _calculate_energy(self):
        """Calculate energy per spin for the current state of the XY system.
        """
        energy = 0
        for x in range(self.length_xy):
            for y in range(self.length_xy):
                theta = self.spins[x, y]

                # avoid double counting so only right and up
                energy -= np.cos(theta - self.spins[x, (y+1) % self.length_xy]) # up neighbour
                energy -= np.cos(theta - self.spins[(x+1) % self.length_xy, y]) # right neighbour
        energy_per_spin = energy / self.length_xy**2
        return energy_per_spin
    
    
    def _calculate_magnetization(self, abs=False):
        """Calculate magnetization per spin m = M/N^2 where M = sum(spins). 
        If abs=True, returns the norm of the m vector, resulting in a 1D array of magn_hist.
        If abs=False, returns the x and y components of the m vector, resulting in a 2D array magn_hist
        """
        m = None
        if abs == True:
            M = np.abs(np.sum(np.exp(1j * self.spins)))
            m = M / self.length_xy**2
            return m
        elif abs == False:
            Mx = np.sum(np.cos(self.spins))
            My = np.sum(np.sin(self.spins))
            return np.array([Mx, My]) / self.length_xy**2
        else:
            raise ValueError('Please choose abs=Bool')
        
    def _calculate_vortex_density(self):
        """Calculate vortex density from a grid of spins. 
        Uses the find_vortices function from utils to identify vortices and antivortices.

        Parameters
        ----------
        none

        Returns
        -------
        vortex_density : float
            Vortex density, i.e. the number of vortices per spin.
        n_vortices : int
            Number of vortices.
        n_antivortices : int
            Number of antivortices.
        """
        vortices = find_vortices(self.spins)
        n_vortices = sum(v[2] == 1 for v in vortices)
        n_antivortices = sum(v[2] == -1 for v in vortices)
        vortex_density = (n_vortices + n_antivortices) / self.length_xy**2
        return vortex_density, n_vortices, n_antivortices

    # def _calculate_autocorrelation(self, t):
    #     pass
    
    # def calculate_specific_heat(self):
    #     pass
    
    # def calculate_susceptibility(self):
    #     pass

