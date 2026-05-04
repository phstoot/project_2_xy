import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas as pd
from src import simul
from src import utils
from src import analysis


# script draft to calculate mean and std of four observables as a function of temp.
# for this we use the low resolution data that has run for more than 20 blocks of 16tau length
# observables: 
# magn per spin
# E per spin
# magn susceptibility per spin
# specific heat per spin

T = 0.5
ic = 'cold'

data_full = np.load(f'data/low_res/spins_50_T_{T}_{ic}.npy')
# just like in tau estimate (autocorrelation script), we discard a burn in time of 20000 spins for the low temp regime. 
# with a sample interval of 80, this comes down to the first 250 entries.
data = data_full[250:]
tau = 5 # for 0.5, tau = 400 so combine with sample interval

# then divide the data up into blocks of 16 times tau
block_size = tau * 16
n_blocks = len(data) // block_size
trimmed = data[:n_blocks * block_size]
blocks = trimmed.reshape(n_blocks, block_size, 50, 50) 


# wait, energy should not go in blocks. We should do that regularly like magnetization with high_res, because it is not a dynamic quantity.
# only the last two observables should go in block methods

for block in blocks:
    energy_block = []
    for state in block:
        energy_block.append(analysis.energy(state, 50))
    energy_mean = 