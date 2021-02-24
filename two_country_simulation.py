# This script simulates model by Wanner and Watabe (2021)
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import WWmodels

#%% Simulating scenario with solving a model
# 0. North and South
# 1. HFDI or VFDI 
# 2. trade liberalization / investment liberalization
# 3. Potential decomposition

# Set elasticities
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 
L = [1,1]

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
    tau_HFDI = np.empty((N,N,N))
    tau_VFDI = np.empty((N,N,N))
    tau_HFDI[:,:], tau_VFDI[:,:] = np.nan,np.nan
    
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if HQ == PR and PR != DE:
            tau_HFDI[HQ,PR,DE] = 1.05
            tau_VFDI[HQ,PR,DE] = 1.05
        if HQ != PR and PR == DE:
            tau_HFDI[HQ,PR,DE] = tau_il
            tau_VFDI[HQ,PR,DE] = np.nan
        if HQ != PR and HQ == DE:
            tau_HFDI[HQ,PR,DE] = np.nan
            tau_VFDI[HQ,PR,DE] = tau_il
    
    # Kill South to North affiliates
    tau_HFDI[1,0,:] = np.nan
    tau_VFDI[1,0,:] = np.nan    
    return tau_HFDI,tau_VFDI

# Calculate the step
HFDI_path,VFDI_path = np.empty((0,2*N)),np.empty((0,2*N))

trange = np.arange(1,1.5,0.05)
for t in trange:
    tau_HFDI,tau_VFDI = calc_tau(t)
    HFDI = WWmodels.Model_single_industry(N=2,theta=theta,rho=0.55,tau=tau_HFDI,L=L,a=a,g=g)
    VFDI = WWmodels.Model_single_industry(N=2,theta=theta,rho=0.55,tau=tau_VFDI,L=L,a=a,g=g)
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

# Plot the figure
fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
sns.set_style("whitegrid")
line_plot_HFDI = sns.lineplot(data=df_HFDI,ax=axs[0])
axs[0].set_title("HFDI")
axs[0].set_ylabel("emission")
axs[0].set_xlabel("tau")
line_plot_VFDI = sns.lineplot(data=df_VFDI,ax=axs[1])
axs[1].set_title("VFDI")
axs[1].set_xlabel("tau")

#figure_VFDI = line_plot_VFDI.get_figure()
#figure_VFDI.savefig("VFDI.png")
#sns.set_style("whitegrid")
#figure_HFDI = line_plot_HFDI.get_figure()
fig.savefig("./figures/two_country_simulation.png")

print("Kinda done")

#%% Simulating scenario using exact hat-algebra
# Set elasticities
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 

# Let's first simulate the X
X_HFDI,X_VFDI = np.zeros((N,N,N)),np.zeros((N,N,N))
# Assymetric case for HFDI
X_HFDI[0,0,0] = 2
X_HFDI[0,0,1] = 1
X_HFDI[0,1,0] = 0
X_HFDI[0,1,1] = 1
X_HFDI[1,1,0] = 1
X_HFDI[1,1,1] = 1
# Assymetric case for VFDI
X_VFDI[0,0,0] = 2
X_VFDI[0,0,1] = 1
X_VFDI[0,1,0] = 1
X_VFDI[0,1,1] = 0
X_VFDI[1,1,0] = 0
X_VFDI[1,1,1] = 2

"""
for HQ,PR,DE in np.ndindex((N,N,N)): 
    if HQ == PR and PR != DE:
        # Export from headquarters
        X_HFDI[HQ,PR,DE] = 1
        X_VFDI[HQ,PR,DE] = 0
    if HQ != PR and PR == DE:
        # Horizontal FDI
        X_HFDI[HQ,PR,DE] = 1
        X_VFDI[HQ,PR,DE] = 0
    if HQ != PR and HQ == DE:
        # Vertical FDI
        X_HFDI[HQ,PR,DE] = 0
        X_VFDI[HQ,PR,DE] = 1
    if HQ == PR and PR == DE:
        X_HFDI[HQ,PR,DE] = 2
        X_VFDI[HQ,PR,DE] = 3 
"""

# Set emission
Z_HFDI,Z_VFDI = copy.deepcopy(X_HFDI),copy.deepcopy(X_VFDI)
Z_HFDI[1,:,:] = Z_HFDI[1,:,:] * 1.5
Z_VFDI[1,:,:] = Z_VFDI[1,:,:] * 1.5
E_HFDI,E_VFDI = copy.deepcopy(X_HFDI),copy.deepcopy(X_VFDI)

# Set tau
def calc_tauhat(tauhat_il):
    """
    This function simply calculates tau
    given for some parameter tau_dif for both
    HFDI and VFDI 
    """
    tau_HFDI = np.ones((N,N,N))
    tau_VFDI = np.ones((N,N,N))
    
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if HQ != PR:
            # Multinational production
            tau_HFDI[HQ,PR,DE] = tauhat_il
            tau_VFDI[HQ,PR,DE] = tauhat_il
    return tau_HFDI,tau_VFDI

# Calculate the step
HFDI_path,VFDI_path = np.empty((0,2*N)),np.empty((0,2*N))

trange = np.arange(0.75,1.25,0.05)
for t in trange:
    tauhat_HFDI,tauhat_VFDI = calc_tauhat(t)
    HFDI = WWmodels.Model_single_industry(N=2,theta=theta,rho=0.55,X=X_HFDI,Z=Z_HFDI,E=E_HFDI,L=L,a=a,g=g)
    VFDI = WWmodels.Model_single_industry(N=2,theta=theta,rho=0.55,X=X_VFDI,Z=Z_VFDI,E=E_VFDI,L=L,a=a,g=g)
    _,_,_,_,_,X1_HFDI,Z1_HFDI,E1_HFDI = HFDI.exacthatalgebra(tauhat_HFDI)
    _,_,_,_,_,X1_VFDI,Z1_VFDI,E1_VFDI = VFDI.exacthatalgebra(tauhat_VFDI)
    HFDI_column = np.concatenate([np.sum(Z1_HFDI,axis=(0,2)),np.sum(E1_HFDI,axis=(0,2))])
    VFDI_column = np.concatenate([np.sum(Z1_VFDI,axis=(0,2)),np.sum(E1_VFDI,axis=(0,2))])
    HFDI_path = np.vstack((HFDI_path,HFDI_column))
    VFDI_path = np.vstack((VFDI_path,VFDI_column))

# Put this into dataframe
column_names_HFDI = ["Z_North_HFDI","Z_South_HFDI","E_North_HFDI","E_South_HFDI"]
column_names_VFDI = ["Z_North_VFDI","Z_South_VFDI","E_North_VFDI","E_South_VFDI"]
df_HFDI = pd.DataFrame(data=HFDI_path,columns = column_names_HFDI,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_path,columns = column_names_VFDI,index=trange)
df_twocountry = pd.merge(left=df_HFDI,right=df_VFDI,how="inner",left_index=True,right_index=True)
# Plot the figure
fig,ax =plt.subplots()
sns.set_style("whitegrid")
line_plot = sns.lineplot(data=df_twocountry,ax=ax)
ax.set_ylabel("emission")
ax.set_xlabel("tau")
fig.savefig("./figures/two_country_simulation_ha.png")
#axs[0].set_xlabel("tau")
#fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
#sns.set_style("whitegrid")
#line_plot_HFDI = sns.lineplot(data=df_HFDI,ax=axs[0])
#axs[0].set_title("HFDI")
#axs[0].set_ylabel("emission")
#axs[0].set_xlabel("tau")
#line_plot_VFDI = sns.lineplot(data=df_VFDI,ax=axs[1])
#axs[1].set_title("VFDI")
#axs[1].set_xlabel("tau")
print("Kinda done")
