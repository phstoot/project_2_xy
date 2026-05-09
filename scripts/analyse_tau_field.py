###############################################
# EXTERNAL FIELD: estimate correlation time tau
# This script loads the simulation data, 
# processes it and estimates tau,
# creates plots and stores results.
###############################################
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas as pd
import glob
import json
from src import simul
from src import utils
from src.analysis import autocorrelation, correlation_time, autocorrelation_curve

root = utils.find_project_root()

# change to control interactively
batch = 0


##############################################
# Magnetization plot (to check for convergence)
################################################

def plot_magnetization(data_dir, temps, discard, ics):
    """Quick plots of magnetization history at a certain temp to check for convergence
    """
    for j in range(len(temps)):
        fig1, ax = plt.subplots(figsize=(7,2.5), layout='constrained')
        ax.set_title(f'Magnetization per spin, T={temps[j]}')
        ax.set_xlabel('t (lattice sweeps)')
        ax.set_ylabel(r'$\langle|m|\rangle$ ($|M|/N^2$)')
        for ic in ics:
            try:
                magn_hist = np.load(data_dir/f'field_magn_50_T_{temps[j]}_{ic}.npy')
            except:
                raise RuntimeError('No simulation results found.')
            
            m = np.linalg.norm(magn_hist, axis=1)
            ax.plot(m, label=f'{ic} start', alpha=0.7, linewidth=0.8)
            ax.set_xlim(0,len(m))
            ax.set_ylim(0, 1)

            ax.axvline(discard[j], color='black')
            # place text boxes in upper left in axes coords with plot info
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # textstr = fr'$T$ = {temps[j]}, $J$ = 1, $k_B$ = 1, burnin = {discard[j]}'
            # axs[0].text(0.98, 0.95, textstr, fontsize=11,
            #         verticalalignment='top', horizontalalignment='right', transform= axs[0].transAxes, bbox=props)
        ax.legend()
        # plt.savefig(f'results/magn_50_T_{temps[j]}.pdf')
        plt.show()
        plt.close('all')


###############################################
#   autocorrelation plot + tau fitting
###############################################

def tau_pipeline(data_dir, results_dir, temps, discard, ics, plot=False):
    """The main pipeline for calculating the correlation time of the magnetization autocorrelation function.
    It makes use of several functions from the analysis module and scipy.optimize.curve_fit to find best
    fitting values for tau given the magnetization data. After calculating tau for every temp, it outputs
    the results to a predefined txt file in results.

    Parameters
    ----------
    plot : Bool, optional
        whether to plot the autocorrelation function and fit, by default False
    """
    # initialize dicts
    taus = {}
    taus['cold'] = []
    taus['hot'] = []
    tau_vars = {}
    tau_vars['hot'] = []
    tau_vars['cold'] = []
    for j in range(len(temps)):
        axs=[] # for pylance
        if plot==True:
            fig, axs = plt.subplots(2, 1, sharex=True)
            fig.suptitle(r'Autocorrelation function $\chi(t)$')
            fig.subplots_adjust(hspace=0)

        for i, ic in enumerate(ics):
            try:
                data = np.load(data_dir/f'field_magn_50_T_{temps[j]}_{ic}.npy')
            except:
                raise RuntimeError('No simulation results found.')
            # calculate autocorrelation function
            m_abs = np.linalg.norm(data, axis=1)
            corr = autocorrelation(m_abs, discard[j])
            corr = corr / corr[0]
            tau_guess = correlation_time(corr)

            # prepare fit
            max_lim = int(3*tau_guess)
            t_fit = np.arange(max_lim) # we fit from t = 0 at the end of burn in to some range after
            t_curve = np.arange(0, len(corr)) 
            t_plot = np.arange(discard[j], discard[j] + len(corr)) # translate axis for plotting to account for burn in
            print(f"Fitting tau for T={temps[j]}, {ic} start, guess = {tau_guess:.2f}...")


            # perform fit
            popt, pcov = sco.curve_fit(
                autocorrelation_curve,
                t_fit,
                corr[:max_lim],
                p0=(1, tau_guess+10),
                # bounds=([0.7, 0.1*tau_guess], [1.2, 5*tau_guess])
            )
            fittedcurve = autocorrelation_curve(t_curve, *popt)
            
            # save parameter fit
            taus[f'{ic}'].append(popt[1])
            tau_vars[f'{ic}'].append(pcov[1,1])
            
            # plot
            if plot == True:
                axs[i].plot(t_plot, corr, label=f'{ic} start') 
                axs[i].plot(t_plot, fittedcurve, label=fr'$\tau = ${popt[1]:.3g} ', linestyle='--')
                axs[i].set_title(r'Autocorrelation function $\chi(t)$')
            
                axs[i].set_ylabel(r'$\chi(t)$')
                axs[1].set_xlabel(r't (lattice sweeps)')

                # axs[i].set_xlim(discard[j], discard[j]+ 2000)
                axs[i].axhline(0, linestyle='--', color='grey')
            
                axs[i].legend()
                # place text boxes in upper left in axes coords with plot info
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = fr'$\tau$ = {popt[1]:.3g}'
                axs[1].text(0.98, 0.95, textstr, transform=axs[1].transAxes, fontsize=13,
                        verticalalignment='top', horizontalalignment='right', bbox=props)

        if plot==True:
            plt.show()
            plt.close('all')

    with open(results_dir/f"field_tau_results_50_{batch}.txt", "w") as out:
        out.write("temp start tau tau_variance\n")
        for i, T in enumerate(temps):
            for ic in ics:
                out.write(f"{T} {ic} {taus[f'{ic}'][i]} {tau_vars[f'{ic}'][i]}\n")


#################################
# tau-temp plot + final results
#################################

def final_tau():
    """Combine all tau calculations from several batches into one final plot and results file.
    This is the final part of the methods: we take the mean of the sample for every tau, and 
    report the standard deviation as error. 
    """
    files = glob.glob(str(root / "results/high_res*/field_tau_results_50_*.txt"))
    if len(files) == 0:
        raise RuntimeError("No files matched glob pattern.")
    try:
        df = pd.concat([pd.read_csv(f, sep=" ") for f in files])
    except: 
        raise RuntimeError('No correlation times found.')
    
    grouped = df.groupby(["temp"])["tau"].agg(["mean", "std"]).reset_index()
    grouped.to_csv(str(root / "results/field_tau_temp.txt"), sep="\t", index=False)

    tau_final = pd.read_csv(str(root / "results/field_tau_temp.txt"), sep='\t')
    fig = plt.figure(figsize=(7,3.5))
    plt.errorbar(
        tau_final['temp'],
        tau_final['mean'],
        tau_final['std'],
        fmt='o',                 # marker style
        markersize=7,
        color="#000000",         # main color
        ecolor="#000000",        # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9
    )
    plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1)
    plt.ylabel(r'$\tau$ (lattice sweeps)', size=14)
    plt.xlabel(r'$T$', size=14)
    plt.ylim(0, 1.5* max(tau_final['mean']))
    plt.axhline(0, color='grey', linestyle=':', alpha=0.7)
    plt.xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(str(root / "results/field_tau_temp.pdf"))
    plt.show()


#################################
# script main()
#################################


def main(batch):
    # manage paths
    data_dir = root / f"data/high_res_{batch}"
    results_dir = root / f"results/high_res_{batch}"
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

    # check for convergence
    plot_magnetization(data_dir, temps, discard, ics)

    # calculate tau for this batch
    tau_pipeline(data_dir, results_dir, temps, discard, ics, plot=False)


if __name__ == '__main__':
    for batch in range(11,15): # check if same in simulate_tau_field.py
        main(batch)
    final_tau()