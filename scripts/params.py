########################
# 
# This script prepares the json parameter files based on our setup plan
# 
#########################
import json
import numpy as np
import pandas as pd

###############################
# 1) HIGH RES RUN
#
# This run is meant to estimate correlation times.
# High resolution, long enough for good burn in and
# adequate precision of the autocorrelation function
# we don't need a certain number of blocks to sample though,
# so the length of the runs has been determined through 
# quantitative testing of X(t)
#
################################

temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
sweeps = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 10000, 10000, 5000, 5000, 4000, 4000]
discard = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 500, 500, 500, 500, 500]
sample_interval = np.full(14, 1, dtype=int)
N = 50
ics = ['hot', 'cold']

params = {
    "description": "XY simulation of size 50. High res run to estimate correlation time tau. Sweeps and burn-in time manually determined, sample interval = 1, hot+cold",
    "temps": temps,
    "sweeps": sweeps,
    "sample_interval": sample_interval.tolist(),
    "burn_in": discard,
    "N": N,
    "ics": ics
}


with open(f"data/high_res/batch_params.json", "w") as f:
    json.dump(params, f, indent=2)


###############################
# 2) LOW RES RUN
################################

temps = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
taus = pd.read_csv('results/high_res/tau_temp.txt', delimiter='\t')[['temp','mean']]
taus_lowres = taus[taus['temp'].isin(temps)]['mean'].astype('int').reset_index(drop=True)
sweeps = 22 * 16 * taus_lowres
sample_interval = 0.25 * taus_lowres
sample_interval = sample_interval.astype('int').replace(0, 1)
block_size = np.array(((16 * taus_lowres) / sample_interval), dtype=int) # important conversion
discard = 2 * block_size
N = 50
ics = ['hot', 'cold']

params = {
    "description": "XY simulation of size 50. Low res run, tau extracted from high_res analysis. blocksize = 16*tau, sweeps = 22 blocks, burnin time = 2 blocks. sample interval = 0.25*tau, hot+cold",
    "temps": temps,
    "taus": taus_lowres.tolist(),  # convert from pandas to plain list
    "block_size": block_size.tolist(),
    "sweeps": sweeps.tolist(),
    "sample_interval": sample_interval.tolist(),
    "burn_in": discard.tolist(),
    "N": N,
    "ics": ics
}

with open("data/low_res/batch_params.json", "w") as f:
    json.dump(params, f, indent=2)
