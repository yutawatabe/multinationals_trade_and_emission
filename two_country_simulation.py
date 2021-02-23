# This script simulates model by Wanner and Watabe (2021)
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import WWmodels

#%% Simulating scenario
# This should be better done in exact hat-algebra
# 0. North and South
# 1. HFDI or VFDI 
# 2. trade liberalization / investment liberalization
# 3. Potential decomposition

# Set elasticities
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 
L = [2,1]

# Set tau (but normalize things so that domestic tau will be tau)

# Set a (So that South is more dirty)
a = np.ones((N,N,N))
a[1,:,:] = 1.5
g = np.ones((N,N,N))

# Set tau
def calc_tau(tau_il):
    """
    This function simply calculates tau
    given for some parameter tau_dif for both
    HFDI and VFDI 
    """
    tau_HFDI = np.ones((N,N,N))
    tau_VFDI = np.ones((N,N,N))
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if HQ == PR and PR != DE:
            tau_HFDI[HQ,PR,DE] = tau_il
            tau_VFDI[HQ,PR,DE] = tau_il
        if HQ != PR and PR == DE:
            tau_HFDI[HQ,PR,DE] = 1.05
            tau_VFDI[HQ,PR,DE] = np.nan
        if HQ != PR and HQ == DE:
            tau_HFDI[HQ,PR,DE] = np.nan
            tau_VFDI[HQ,PR,DE] = 1.05
    return tau_HFDI,tau_VFDI

# Calculate the step

HFDI_path,VFDI_path = np.empty((0,2*N)),np.empty((0,2*N))

trange = np.arange(2,1,-0.05)
for t in trange:
    tau_HFDI,tau_VFDI = calc_tau(t)
    HFDI = WWmodels.Model_single_industry(N=2,tau=tau_HFDI,L=L,a=a,g=g)
    VFDI = WWmodels.Model_single_industry(N=2,tau=tau_VFDI,L=L,a=a,g=g)
    HFDI.solve()
    VFDI.solve()
    HFDI_column = np.concatenate([np.sum(HFDI.Z,axis=(0,2)),np.sum(HFDI.E,axis=(0,2))])
    VFDI_column = np.concatenate([np.sum(VFDI.Z,axis=(0,2)),np.sum(VFDI.E,axis=(0,2))])
    HFDI_path = np.vstack((HFDI_path,HFDI_column))
    VFDI_path = np.vstack((VFDI_path,VFDI_column))

# Put this into dataframe
column_names = ["Z_North","Z_South","E_North","E_South"]
df_HFDI = pd.DataFrame(data=HFDI_path,columns = column_names,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_path,columns = column_names,index=trange)


sns.set_style("whitegrid")
line_plot_VFDI = sns.lineplot(data=df_VFDI)
figure_VFDI = line_plot_VFDI.get_figure()
figure_VFDI.savefig("VFDI.png")

#sns.set_style("whitegrid")
#line_plot_HFDI = sns.lineplot(data=df_HFDI)
#figure_HFDI = line_plot_HFDI.get_figure()
#figure_HFDI.savefig("HFDI.png")

print("Kinda done")