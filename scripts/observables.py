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

# script draft to calculate mean and std of four observables as a function of temp.
# observables: 
# magn per spin
# E per spin
# magn susceptibility per spin
# specific heat per spin

################################################
# 1: for the first two, we use the high resolution data
################################################

temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
discard = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 500, 500, 500, 500, 500]
ics = ['hot', 'cold']
taus = pd.read_csv('results/high_res/tau_temp.txt', delimiter='\t')[['temp','mean']]


def main():
    # loop through runs
    for ic in ics:
        e_means = []
        e_stds = []
        m_means = []
        m_stds = []
        for i, T in enumerate(temps):
            e_full = np.load(f'data/high_res_0/energy_50_T_{T}_{ic}.npy')
            e_data = e_full[discard[i]:]
            e_data = e_data
            e_means.append(np.mean(e_data))
            e_stds.append(analysis.independend_std(e_data, int(taus['mean'][i])))

            m_full = np.load(f'data/high_res_0/magn_50_T_{T}_{ic}.npy')
            m_data = m_full[discard[i]:]
            m_means.append(np.mean(m_data))
            m_stds.append(analysis.independend_std(m_data, int(taus['mean'][i])))
            print(T, np.mean(m_data))
        e_df = pd.DataFrame({
        "temp": temps,
        "mean": e_means,
        "std": e_stds
        })
        e_df.to_csv(f'results/high_res/e_temp_{ic}.txt', sep="\t", index=False)
        
        m_df = pd.DataFrame({
        "temp": temps,
        "mean": m_means,
        "std": m_stds
        })
        m_df.to_csv(f'results/high_res/m_temp_{ic}.txt', sep="\t", index=False)


if __name__ == '__main__':
    main()


e_df = pd.read_csv('results/high_res/e_temp_cold.txt', sep='\t')
fig = plt.figure(figsize=(6,3))
plt.errorbar(
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
plt.ylabel(r'$e$ ($E/N^2$)')
plt.xlabel(r'$T$ (unitless)')
plt.xlim(0.4, 2.6)
plt.tight_layout()
plt.savefig('e_temp.pdf')
plt.show()

m_df = pd.read_csv('results/high_res/m_temp_cold.txt', sep='\t')
fig = plt.figure(figsize=(6,3))
plt.errorbar(
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
plt.ylabel(r'$m$ ($M/N^2$)')
plt.xlabel(r'$T$ (unitless)')
plt.xlim(0.4, 2.6)
plt.tight_layout()
plt.savefig('m_temp.pdf')
plt.show()



# combine in one report plot

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

# prepare parameters
with open("data/low_res_1/batch_params.json") as f:
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


batch = 1

for ic in ics:
    susc_means = []
    susc_err = []
    heat_means = []
    heat_err = []
    for i, T in tqdm(enumerate(temps)):
        magn_full = np.load(f'data/low_res_{batch}/magn_50_T_{T}_{ic}.npy') * 50 # convert to total
        energy_full = np.load(f'data/low_res_{batch}/energy_50_T_{T}_{ic}.npy') * 2500 # convert to total

        # discard burn in
        magn_arr = magn_full[discard[i]:]
        energy_arr = energy_full[discard[i]:]
        if len(magn_arr) != len(energy_arr):
            raise UserWarning('different lengths')
        
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

        # print('m', np.shape(m_blocks))
        # print('e', np.shape(e_blocks))
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
    susc_df.to_csv(f'results/low_res_{batch}/susc_temp_{ic}.txt', sep="\t", index=False)

    heat_df = pd.DataFrame({
        "temp": temps,
        "mean": heat_means,
        "err": heat_err
        })
    heat_df.to_csv(f'results/low_res_{batch}/heat_temp_{ic}.txt', sep="\t", index=False)





susc_df = pd.read_csv(f'results/low_res_{batch}/susc_temp_hot.txt', sep='\t')
# plt.style.use("seaborn-v0_8-whitegrid")  # clean base style
fig = plt.figure(figsize=(6,3))
# plt.scatter(df['temp'], df['tau'])
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
# plt.tick_params(direction='in', length=5, width=1)
plt.ylabel(r'$\chi_m$ ($N^{-2}$ K$^{-1}$)')
plt.xlabel(r'$T$ (unitless)')
# plt.ylim(-100, 1300)
# plt.axhline(0, color='grey', alpha=0.7)
plt.xlim(0.4, 2.6)
plt.tight_layout()
plt.savefig(f'results/low_res_{batch}/susc_temp_hot.pdf')
plt.show()

heat_df = pd.read_csv(f'results/low_res_{batch}/heat_temp_cold.txt', sep='\t')
# plt.style.use("seaborn-v0_8-whitegrid")  # clean base style
fig = plt.figure(figsize=(6,3))
# plt.scatter(df['temp'], df['tau'])
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
# plt.ylim(-100, 1300)
# plt.axhline(0, color='grey', alpha=0.7)
plt.xlim(0.4, 2.6)
plt.tight_layout()
# plt.savefig(f'results/low_res_{batch}/heat_temp_hot.pdf')
plt.show()




##### combine batches for magnetic susceptibility

batches = [0, 1, 2]
for ic in ics:
    dfs = []
    for b in batches:
        df = pd.read_csv(f'results/low_res_{b}/susc_temp_{ic}.txt', sep='\t')
        dfs.append(df)

    combined = pd.concat(dfs)
    final = combined.groupby('temp')['mean'].mean().reset_index()
    # combine errors in quadrature: sqrt(sum of variances) / n_batches
    err_combined = combined.groupby('temp')['err'].apply(
        lambda x: np.sqrt(np.sum(x**2)) / len(x)
    ).reset_index()
    final['err'] = err_combined['err']
    final.to_csv(f'results/susc_temp_final_{ic}.txt', sep='\t', index=False)


    # plt.style.use("seaborn-v0_8-whitegrid")  # clean base style
    fig = plt.figure(figsize=(6,3))
    plt.errorbar(
        final['temp'],
        final['mean'],
        final['err'],
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
    # plt.tick_params(direction='in', length=5, width=1)
    plt.ylabel(r'$\chi_m$ ($N^{-2}$ K$^{-1}$)')
    plt.xlabel(r'$T$ (unitless)')
    # plt.ylim(-100, 1300)
    # plt.axhline(0, color='grey', alpha=0.7)
    plt.xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(f'results/susc_temp_final_{ic}.pdf')
    plt.show()

# but also combine them into one final plot
ics = ['hot', 'cold']
batches = [0, 1, 2]

all_dfs = []
for ic in ics:
    for b in batches:
        df = pd.read_csv(f'results/low_res_{b}/susc_temp_{ic}.txt', sep='\t')
        all_dfs.append(df)

combined = pd.concat(all_dfs)

def combine_measurements(group):
    means = group['mean'].values
    errs = group['err'].values
    n = len(means)
    
    # combined mean
    grand_mean = np.mean(means)
    
    # statistical: propagated error on the mean
    stat_err = np.sqrt(np.sum(errs**2)) / n
    
    # systematic: sample std of the 6 means
    syst_err = np.std(means, ddof=1) / np.sqrt(n)
    
    # combined in quadrature
    total_err = np.sqrt(stat_err**2 + syst_err**2)
    
    return pd.Series({'mean': grand_mean, 'err': total_err})

final = combined.groupby('temp').apply(combine_measurements).reset_index()
final.to_csv(f'results/susc_temp_final.txt', sep='\t', index=False)


heat_df = pd.read_csv(f'results/low_res_{batch}/heat_temp_cold.txt', sep='\t')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,5), sharex=True)
ax1.errorbar(
    final['temp'],
    final['mean'],
    final['err'],
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


###################################################

# # again we only need the energy array, not the spins. So we quickly write energy to arrays using numba, work from there on
def warmup():
    dummyspins = np.zeros((50,50))
    analysis.energy(dummyspins, 50)


def main2():
    # warmup()
    for ic in ics:
        for T in temps:
            e_hist = []
            data_full = np.load(f'data/low_res/spins_50_T_{T}_{ic}.npy')

            # for state in tqdm(data_full):
            #     energy = analysis.energy(state, 50)
            #     e_hist.append(energy)
            onespin = data_full[:,5,5]
            twospin = data_full[:,9,3]
            plt.plot(onespin)
            plt.plot(twospin)
            plt.title(f'E, T={T}, ic = {ic}')
            plt.show()
            # np.save(f'data/low_res/energy_T_{T}_{ic}', np.array(e_hist))



if __name__ == '__main__':
    main2()

