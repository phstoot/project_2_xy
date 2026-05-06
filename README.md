### Computational Physics B

# Monte Carlo Simulation of the XY Model

---

## Overview

This project implements a Monte Carlo simulation of the 2D XY model on a square lattice. The goal is to study thermodynamic observables such as magnetization, energy, susceptibility, and correlation time using the Metropolis Hastings algorithm.

The code is structured as a small Python package with separate modules, along with scripts to reproduce the results used in the report.

---

## Installation

### For development (collaborators)

Install in editable mode inside your environment:

```bash
pip install -e .
```

### For assessment / general use

Install in a your dedicated environment:

```bash
pip install .
```

---

## Model and Simulation Details

We simulate the classical 2D XY model with the following choices:

* Coupling constant: ( $J = 1$ )
* Boltzmann constant: ( $k_B = 1$ )
* External field: none
* Lattice size: ( 50 $\times$ 50 )
* Boundary conditions: periodic
* Update method: Metropolis Hastings Monte Carlo
* Spin representation: angles ( $\theta \in [0, 2\pi]$ )

### Implementation and reproducability notes

* The simulation is accelerated using **Numba JIT compilation**
* Random updates are performed per lattice site (Monte Carlo sweeps)
* 'start' corresponds to a 'cold' or 'hot' start, respectively starting with all spins aligned or a random configuration.
* Magnetization and energy are recorded over time and post-processed using blocking/statistical analysis
* Individual states can be visualised using instance.static_image()
* Due to the stochastic nature of the Monte Carlo approach, results will not be reproducible to the exact value. 
* Final reported results are obtained using multiple independent batches. This setup is handled respectively by utils.high_res_setup() and utils.low_res_setup(), both are explained in their docstrings.

---

## Project Structure

```text
project/
│
├── src/
│   ├── simul.py        # Monte Carlo simulation (MonteCarlo_XY)
│   └── analysis.py     # Observable calculations and statistics
│   └── utils.py        # Observable calculations and statistics
│
├── scripts/
│   ├── simulate_*.py # Generate raw simulation data
│   └── analyse_*.py  # Process data and produce plots
│
├── data/             # Output data (generated)
├── results/          # post-processed results (generated)
└── README.md
```

---

## Reproducing Results

To reproduce the results shown in the report:

### 1. Run simulations

```bash
python scripts/simulate_*.py
```

This generates raw data in the `data/` directory.

### 2. Run analysis

```bash
python scripts/analyse_*.py
```

This processes the data and produces final results and plots in the `results/` directory.

> Note: Full simulations may take significant time depending on the number of sweeps and batches. According to our tests and using the Numba implementation, a full suite consisting of ~10^5 sweeps for 14 temperatures should not take longer than 20 minutes. 

---

## General Usage

The package can also be used outside of the pre-made scripts. See the simple example below:

### Example

```python
from src.simul import MonteCarlo_XY
from src.analysis import autocorrelation, energy, magn_susc

# Initialize simulation
sim = MonteCarlo_XY(length_xy=50, temperature=1, start='cold')

# Run simulation
sim.run(sweeps=1000, store=True, interval=1)

# Access recorded data
magnet_data = sim.magn_hist
energy_data = sim.e_hist

# Display end state
sim.static_image()

# Perform some simple analysis
autocorr = autocorrelation(magn_hist, discard=100)
tau_estimate = correlation_time(autocorr)
chi = magn_susc(magnet_data, temp=1, k_b=1, length_xy=1)
C = energy(energy_data, temp=1, k_b=1, length_xy=1)
```

---

## Authors

Nils Thiessen, 
Philip Stoot

---
