import numpy as np
import matplotlib.pyplot as plt
from src import simul


def autocorrelation(m):
    # slice off equilibration time, for now let's do 500 timesteps
    m = m[250:]
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


temps = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames

for ic in ['hot', 'cold']:
    for T in temps:
        data = np.load(f'data/magn_50_{ic}start_T_{T}.npy')
        corr = autocorrelation(data)
        tau = correlation_time(corr)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', layout='constrained')
        fig.suptitle(f'XY model', weight='bold')
        ax1.plot(data)
        ax1.set_title(f'Magnetization per spin')
        ax1.set_xlabel('t (lattice sweeps)')
        ax1.set_ylabel(r'm ($M/N^2$)')
        ax1.set_xlim(0,len(data))
        ax1.set_ylim(0, 1.1)

        # place text boxes in upper left in axes coords with plot info
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = fr'$T$ = {T}, $J$ = 1, $k_B$ = 1, IC = {ic}'
        ax1.text(0.98, 0.95, textstr, fontsize=11,
                verticalalignment='top', horizontalalignment='right', transform= ax1.transAxes, bbox=props)

        ax2.plot(np.arange(250,2000, 1), corr)
        ax2.set_title(r'Autocorrelation function $\chi(t)$')
        ax2.set_ylabel(r'$\chi(t)$')
        ax2.set_xlabel(r't (lattice sweeps)')
        ax2.axhline(0, linestyle='--', color='grey')
        ax2.axvline(250, color='grey')
        
        # place text boxes in upper left in axes coords with plot info
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = fr'$\tau$ = {tau:.3g}'
        ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=13,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig(f'results/mag_x_50_{ic}start_T_{T}.pdf')
        # plt.show()
        plt.close('all')

