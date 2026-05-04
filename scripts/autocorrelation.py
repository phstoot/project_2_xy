import numpy as np
import matplotlib.pyplot as plt
from src import simul
import scipy.optimize as sco
import pandas as pd
import glob
from src.analysis import autocorrelation, correlation_time, autocorrelation_curve

temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5] # prevent weird floating point error in filenames
discard = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 500, 500, 500, 500, 500]
ics = ['hot', 'cold']

# change to control
batch = 3


##############################################
# Magnetization plot (check for convergence)
################################################

for j in range(len(temps)):
    fig1, ax = plt.subplots(figsize=(7,2.5), layout='constrained')
    ax.set_title(f'Magnetization per spin, T={temps[j]}')
    ax.set_xlabel('t (lattice sweeps)')
    ax.set_ylabel(r'm ($M/N^2$)')
    for i in [0,1]:
        data = np.load(f'data/high_res_{batch}/magn_50_T_{temps[j]}_{ics[i]}.npy')
        
        ax.plot(data, label=f'{ics[i]} start', alpha=0.7, linewidth=0.8)
        ax.set_xlim(0,len(data))
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
taus = {}
taus['cold'] = []
taus['hot'] = []
tau_vars = {}
tau_vars['hot'] = []
tau_vars['cold'] = []
for j in range(len(temps)):
    # fig, axs = plt.subplots(2, 1, sharex=True)
    # fig.suptitle(r'Autocorrelation function $\chi(t)$')
    # fig.subplots_adjust(hspace=0)

    for i in [0,1]:
        data = np.load(f'data/high_res_{batch}/magn_50_T_{temps[j]}_{ics[i]}.npy')

        corr = autocorrelation(data, discard[j])
        corr_norm = corr / corr[0]
        tau_guess = correlation_time(corr)
        print('guess for tau:', tau_guess)
        max_lim = int(3*tau_guess)
        t_fit = np.arange(max_lim) # we fit from t = 0 at the end of burn in to some range after
        t_curve = np.arange(0, len(corr)) 
        t_plot = np.arange(discard[j], discard[j] + len(corr)) # translate axis for plotting to account for burn in

        # perform fit
        popt, pcov = sco.curve_fit(
            autocorrelation_curve,
            t_fit,
            corr_norm[:max_lim],
            p0=(1, tau_guess+10),
            # bounds=([0.7, 0.1*tau_guess], [1.2, 5*tau_guess])
        )
        fittedcurve = autocorrelation_curve(t_curve, *popt)
        
        # save parameter fit
        print('\npopt:', popt, '\npcov:', pcov)
        # np.savetxt(f'results/high_res/popt_50_T_{temps[j]}_{ics[i]}.txt', popt)
        # np.savetxt(f'results/high_res/pcov_50_T_{temps[j]}_{ics[i]}.txt', pcov)
        taus[f'{ics[i]}'].append(popt[1])
        tau_vars[f'{ics[i]}'].append(pcov[1,1])

        
        # plot
        # axs[i].plot(t_plot, corr_norm, label=f'{ics[i]} start')
        # axs[i].plot(t_plot, fittedcurve, label=fr'$\tau = ${popt[1]:.3g} ', linestyle='--')
        # # axs[i].set_title(r'Autocorrelation function $\chi(t)$')
    
        # axs[i].set_ylabel(r'$\chi(t)$')
        # axs[1].set_xlabel(r't (lattice sweeps)')

        # # axs[i].set_xlim(discard[j], discard[j]+ 2000)
        # axs[i].axhline(0, linestyle='--', color='grey')
    
        # axs[i].legend()
    #     # # place text boxes in upper left in axes coords with plot info
    #     # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #     # textstr = fr'$\tau$ = {tau:.3g}'
    #     # axs[1].text(0.98, 0.95, textstr, transform=axs[1].transAxes, fontsize=13,
    #     #         verticalalignment='top', horizontalalignment='right', bbox=props)

    # plt.savefig(f'results/crit/corr_50_T_{temps[j]}.pdf')
    # plt.show()
    plt.close('all')

with open(f"results/high_res/tau_results_50_{batch}.txt", "w") as out:
    out.write("temp start tau tau_variance\n")
    for i, T in enumerate(temps):
        for ic in ['hot', 'cold']:
            # x0, tau = np.loadtxt(f'results/high_res/popt_50_T_{T}_{ic}.txt')
            # pcov = np.loadtxt(f"results/high_res/pcov_50_T_{T}_{ic}.txt")
            # tau_var = pcov[1, 1]

            out.write(f"{T} {ic} {taus[f'{ic}'][i]} {tau_vars[f'{ic}'][i]}\n")


#################################
# tau-temp plot
#################################

# data = pd.read_csv(f"results/high_res/tau_results_50_{batch}.txt", sep=" ")
# # data = pd.read_csv(f"results/high_res/tau_results_50_2.txt", sep=" ")
# hot = data[data["start"] == "hot"]
# cold = data[data["start"] == "cold"]


# plt.errorbar(
#     hot["temp"],
#     hot["tau"],
#     hot["tau_variance"]**0.5,
#     label='hot'
# )
# plt.errorbar(
#     cold["temp"],
#     cold["tau"],
#     cold["tau_variance"]**0.5,
#     label='cold'
# )

# plt.legend()
# plt.show()


files = glob.glob("results/high_res/tau_results_50_*.txt")
df = pd.concat([pd.read_csv(f, sep=" ") for f in files])
grouped = df.groupby(["temp"])["tau"].agg(["mean", "std"]).reset_index()
grouped.to_csv("results/high_res/tau_temp.txt", sep="\t", index=False)

# plt.style.use("seaborn-v0_8-whitegrid")  # clean base style
fig = plt.figure(figsize=(6,3))
# plt.scatter(df['temp'], df['tau'])
plt.errorbar(
    grouped['temp'],
    grouped['mean'],
    grouped['std'],
    fmt='o',                 # marker style
    markersize=5,
    color="#000000",         # main color
    ecolor="#000000",        # lighter errorbar color
    elinewidth=1.2,
    capsize=3,
    capthick=1,
    linestyle='--',           # connect points
    linewidth=1,
    alpha=0.9
)
# Ticks
# plt.tick_params(direction='in', length=5, width=1)
plt.ylabel(r'$\tau$ (lattice sweeps)')
plt.xlabel(r'$T$ (unitless)')
plt.ylim(-100, 1300)
plt.axhline(0, color='grey', alpha=0.7)
plt.xlim(0.4, 2.6)
plt.tight_layout()
plt.savefig('tau_temp.pdf')
plt.show()


