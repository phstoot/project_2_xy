import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('project_2_xy/src')
import src.simul as simul
import src.utils as utils



def autocorrelation(m, discard):
    # slice off equilibration time, for now let's do 500 timesteps
    m = m[discard:]
    t_max = len(m)

    X = np.zeros(len(m))
    # we are gonna iterate over all t's within the function
    time = np.arange(len(m))
    for t in range(len(m)):
        factor = 1 / (t_max - t)
        X[t] = factor * np.sum( m[0:(t_max-t)] * m[t:t_max] ) - factor * np.sum( m[0:(t_max-t)] ) * factor * np.sum( m[t:t_max] )

    return X

def correlation_time(x):
    stop_idx = np.where(x <= 0)[0][0]
    tau = np.sum( x[0:stop_idx] / x[0])
    return tau

def autocorrelation_curve(t, x_0, tau):
    x_t = x_0 * np.exp(-t / tau)
    return x_t

def energy(spins, length_xy):
    energy = 0
    for x in range(length_xy):
        for y in range(length_xy):
            theta = spins[x, y]

            # avoid double counting so only right and up
            energy -= np.cos(theta - spins[x, (y+1) % length_xy]) # up neighbour
            energy -= np.cos(theta - spins[(x+1) % length_xy, y]) # right neighbour
    return energy