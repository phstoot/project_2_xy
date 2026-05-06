###############################################
# MILESTONE 2: calculate observables of system:
# This script loads the simulation data, 
# processes it and calculates observables,
# creates plots and stores results.
#
# observables: 
#  - M per spin m
#  - E per spin e
#  - magn susceptibility per spin X_m
#  - specific heat per spin C
###############################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas as pd
from src import simul
from src import utils
from src import analysis
from tqdm import tqdm
import glob
import json


root = utils.find_project_root()

# change to control interactively
batch = 0


################################################
# 1: for the first two, we use the high resolution data
################################################



def e_m_pipeline():
    """Pipeline function to calculate energy and magnetization per spin. Only one batch of simulations is needed.
    """
    # loop through runs
    for ic in ics:
        e_means = []
        e_stds = []
        m_means = []
        m_stds = []
        for i, T in enumerate(temps):
            try:
                e_full = np.load(f'data/high_res_0/energy_50_T_{T}_{ic}.npy')
            except:
                raise RuntimeError('No simulation results (energy) found.')
            e_data = e_full[discard[i]:]
            e_data = e_data
            e_means.append(np.mean(e_data))
            e_stds.append(analysis.independend_std(e_data, int(taus['mean'][i])))

            try:
                m_full = np.load(f'data/high_res_0/magn_50_T_{T}_{ic}.npy')
            except:
                raise RuntimeError('No simulation results (magnetization) found.')
            m_data = m_full[discard[i]:]
            m_means.append(np.mean(m_data))
            m_stds.append(analysis.independend_std(m_data, int(taus['mean'][i])))
            print(T, np.mean(m_data))
        e_df = pd.DataFrame({
        "temp": temps,
        "mean": e_means,
        "std": e_stds
        })
        e_df.to_csv(f'results/high_res_0/e_temp_{ic}.txt', sep="\t", index=False)
        
        m_df = pd.DataFrame({
        "temp": temps,
        "mean": m_means,
        "std": m_stds
        })
        m_df.to_csv(f'results/high_res_0/m_temp_{ic}.txt', sep="\t", index=False)



def e_m_plot():
    # combine in one report plot
    try:
        e_df = pd.read_csv('results/high_res_0/e_temp_cold.txt', sep='\t')
        m_df = pd.read_csv('results/high_res_0/m_temp_cold.txt', sep='\t')
    except:
        raise RuntimeError('no results found')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,5), sharex=True)
    ax1.errorbar(
        m_df['temp'],
        m_df['mean'],
        m_df['std'],
        fmt='o',                 # marker style
        markersize=7,
        color="#e63946",         # main color
        ecolor="#e63946",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    ax1.set_ylabel(r'$\langle |m| \rangle$', size=15)
    ax1.set_xlim(0.4, 2.6)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_label_coords(-0.11, 0.5)
    ax2.errorbar(
        e_df['temp'],
        e_df['mean'],
        e_df['std'],
        fmt='o',                 # marker style
        markersize=7,
        color="#479d2a",         # main color
        ecolor="#63d03fff",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    ax2.set_ylabel(r'$\langle |e| \rangle$', size=15)
    ax2.set_xlabel(r'$T$', size=15)
    ax1.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    ax2.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'results/e_m_T_final.pdf')
    plt.show()







################################################################################
#
# 2: for the last two, we use the low resolution data and the block approach. 
# The data has run for more than 20 blocks of 16tau length
#
################################################################################


def x_c_pipeline():
    for ic in ics:
        susc_means = []
        susc_err = []
        heat_means = []
        heat_err = []
        for i, T in tqdm(enumerate(temps)):
            try:
                magn_full = np.load(data_dir/f'magn_50_T_{T}_{ic}.npy') * 50 # convert to total
                energy_full = np.load(data_dir/f'energy_50_T_{T}_{ic}.npy') * 2500 # convert to total
            except:
                raise RuntimeError('no results found')

            # discard burn in
            magn_arr = magn_full[discard[i]:]
            energy_arr = energy_full[discard[i]:]
            if len(magn_arr) != len(energy_arr):
                raise UserWarning('E and M are different lengths. This probably signals corrupted data.')
            
            #inspect data to check
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(energy_arr)
            ax1.set_title(f'e, T={T}, ic={ic}')
            ax2.plot(magn_arr)
            ax2.set_title(f'm, T={T}, ic={ic}')
            plt.show()

            # then divide the data up into blocks of 16 times tau   
            n_blocks = len(magn_arr) // block_size[i] # both same size

            m_trimmed = magn_arr[:n_blocks * block_size[i]]
            e_trimmed = energy_arr[:n_blocks * block_size[i]]
            
            m_blocks = m_trimmed.reshape(n_blocks, block_size[i], 2) 
            e_blocks = e_trimmed.reshape(n_blocks, block_size[i]) 

            # calculate heat and susc per block
            susc = []
            heat = []
            for block in m_blocks:
                susc.append(analysis.magn_susc(block, T, 1, 50))
            
            for block in e_blocks:
                heat.append(analysis.specific_heat(block, T, 1, 50))

            # now we have a sample of n independent block measurements
            # we can calculate the mean and the std of this sample the usual way
            susc_means.append(np.mean(susc))
            susc_err.append(np.std(susc) / np.sqrt(len(susc)))
            heat_means.append(np.mean(heat))
            heat_err.append(np.std(heat) / np.sqrt(len(heat)))
        susc_df = pd.DataFrame({
            "temp": temps,
            "mean": susc_means,
            "err": susc_err
            })
        susc_df.to_csv(results_dir/f'susc_temp_{ic}.txt', sep="\t", index=False)

        heat_df = pd.DataFrame({
            "temp": temps,
            "mean": heat_means,
            "err": heat_err
            })
        heat_df.to_csv(results_dir/f'heat_temp_{ic}.txt', sep="\t", index=False)


def x_c_plot_batches():
    """Generate plots of the X_m and C results of the individual batches. Not used in final report.
    """
    susc_df = pd.read_csv(results_dir/f'susc_temp_hot.txt', sep='\t')
    fig = plt.figure(figsize=(6,3))
    plt.errorbar(
        susc_df['temp'],
        susc_df['mean'],
        susc_df['err'],
        fmt='o',                 # marker style
        markersize=7,
        color="#492a9d",         # main color
        ecolor="#663ade",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    # Ticks
    plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    plt.ylabel(r'$\chi_m$ ($N^{-2}$ K$^{-1}$)')
    plt.xlabel(r'$T$ (unitless)')
    plt.xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(results_dir/f'susc_temp_hot.pdf')
    plt.show()

    heat_df = pd.read_csv(results_dir/f'heat_temp_cold.txt', sep='\t')
    fig = plt.figure(figsize=(6,3))
    plt.errorbar(
        heat_df['temp'],
        heat_df['mean'],
        heat_df['err'],
        fmt='o',                 # marker style
        markersize=7,
        color="#e88253",         # main color
        ecolor="#e88253",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    # Ticks
    plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    plt.ylabel(r'$C$ ($N^{-2}$ K$^{-2}$)')
    plt.xlabel(r'$T$ (unitless)')
    plt.xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(results_dir/f'heat_temp_hot.pdf')
    plt.show()




##### combine batches for magnetic susceptibility and generate final plot


def final_magn_susc(n_batches):
    """Combine several batches of magnetic susceptibility results in one final set of values
    """

    all_dfs = []
    for ic in ics:
        for b in range(n_batches):
            try:
                df = pd.read_csv(f'results/low_res_{b}/susc_temp_{ic}.txt', sep='\t')
            except:
                raise RuntimeError('No results found')
            all_dfs.append(df)

    combined = pd.concat(all_dfs)
    final = combined.groupby('temp').apply(utils.combine_measurements).reset_index()
    final.to_csv(f'results/susc_temp_final.txt', sep='\t', index=False)


def x_c_plot():
    try:
        susc_df = pd.read_csv(f'results/susc_temp_final.txt', sep='\t')
        heat_df = pd.read_csv(f'results/low_res_1/heat_temp_cold.txt', sep='\t') # for heat, one batch is precise enough
    except:
        raise RuntimeError('no results found')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,5), sharex=True)
    ax1.errorbar(
        susc_df['temp'],
        susc_df['mean'],
        susc_df['err'],
        fmt='o',                 # marker style
        markersize=7,
        color="#492a9d",         # main color
        ecolor="#663ade",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    ax1.set_ylabel(r'$\chi_m$', size=15)
    # plt.xlabel(r'$T$ (unitless)')
    ax1.set_xlim(0.4, 2.6)
    # ax1.set_ylim(bottom=0)
    ax2.errorbar(
        heat_df['temp'],
        heat_df['mean'],
        heat_df['err'],
        fmt='o',                 # marker style
        markersize=7,
        color="#e88253",         # main color
        ecolor="#e88253",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
        )
    ax2.set_ylabel(r'$C$', size=15)
    ax2.set_xlabel(r'$T$', size=15)
    ax2.set_ylim(bottom=0)
    ax1.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    ax2.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'results/heat_susc_T_final.pdf')
    plt.show()




if __name__ == '__main__':
    # e and m -> high res, only one batch needed
    batch = 0
    # manage paths
    data_dir = root / f'data/high_res_{batch}'
    results_dir = root / f'results/high_res_{batch}'
    results_dir.mkdir(parents=True, exist_ok=True)

    # load params
    with open(data_dir/"batch_params.json") as f:
        params = json.load(f)
    print("Parameters description:\n")
    print(params['description'])
    temps = params["temps"]
    sweeps = params["sweeps"]
    sample_interval = params['sample_interval']
    discard = params['burn_in']
    ics = params['ics']
    N = params['N']

    try:
        taus = pd.read_csv('results/tau_temp.txt', delimiter='\t')[['temp','mean']]
    except:
        raise RuntimeError('no results found')

    e_m_pipeline()
    e_m_plot()


    # X_m and C -> low res, several batches are needed
    for batch in range(3):
        # manage paths
        data_dir = root / f'data/low_res_{batch}'
        results_dir = root / f'results/low_res_{batch}'
        results_dir.mkdir(parents=True, exist_ok=True)

        # load params
        with open(data_dir/"batch_params.json") as f:
            params = json.load(f)
        print("Parameters description:\n")
        print(params['description'])
        temps = params["temps"]
        sweeps = params["sweeps"]
        taus = params["taus"]
        block_size = params['block_size']
        sweeps = params['sweeps']
        sample_interval = params['sample_interval']
        discard = params['burn_in']
        ics = params['ics']
        N = params['N']

        x_c_pipeline()
        # x_c_plot_batches() # not used in report

    # outside of batch (several batches handled within func):
    final_magn_susc(3) # check if number of batches correct
    x_c_plot()


