# This script simulates model by Wanner and Watabe (2021)
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import WWmodels

#%% Create some simple function

# Set tau
def set_tau(tau,scenario):
    """
    This function simply calculates path of 
    mp liberalization given for some parameter tau for both
    HFDI and VFDI. Scenario can be mp or trade.
    """
    tau_HFDI = np.ones((N,N,N))
    tau_VFDI = np.ones((N,N,N))
    
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if HQ == PR and PR != DE:
            # Export from headquarters
            if scenario =="mp":
                tau_HFDI[HQ,PR,DE] = 1
                tau_VFDI[HQ,PR,DE] = 1
            if scenario =="trade":
                tau_HFDI[HQ,PR,DE] = tau
                tau_VFDI[HQ,PR,DE] = tau
        if HQ != PR and PR == DE:
            # HFDI production
            if scenario =="mp":
                tau_HFDI[HQ,PR,DE] = tau
                tau_VFDI[HQ,PR,DE] = np.nan
            if scenario =="trade":
                tau_HFDI[HQ,PR,DE] = 1
                tau_VFDI[HQ,PR,DE] = np.nan
        if HQ != PR and HQ == DE:
            # VFDI production
            if scenario =="mp":
                tau_HFDI[HQ,PR,DE] = np.nan
                tau_VFDI[HQ,PR,DE] = tau
            if scenario =="trade":
                tau_HFDI[HQ,PR,DE] = np.nan
                tau_VFDI[HQ,PR,DE] = tau                
    
    # Kill South to North affiliates
    tau_HFDI[1,0,:] = np.nan
    tau_VFDI[1,0,:] = np.nan    
    return tau_HFDI,tau_VFDI

# Set tauhat
def set_tauhat(tauhat,scenario):
    """
    This function simply calculates tauhat
    given for some parameter tau_hat for both
    HFDI and VFDI. Scenario could be trade or mp
    """
    tauhat_HFDI = np.ones((N,N,N))
    tauhat_VFDI = np.ones((N,N,N))    
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if HQ != PR and scenario =="mp":
            # Multinational production
            tauhat_HFDI[HQ,PR,DE] = tauhat
            tauhat_VFDI[HQ,PR,DE] = tauhat
        if PR != DE and scenario =="trade":
            # Multinational production
            tauhat_HFDI[HQ,PR,DE] = tauhat
            tauhat_VFDI[HQ,PR,DE] = tauhat
    return tauhat_HFDI,tauhat_VFDI

# Calculate summary statistics
def calc_aggregate(X,Z,E):
    N = X.shape[0]
    Zl,El = np.sum(Z,axis=(0,2)),np.sum(E,axis=(0,2))
    # Calculate EX,IM,OI,II (Export, import, outward investment, inward investment)
    EX,IM,OI,II = np.zeros((N)),np.zeros((N)),np.zeros((N)),np.zeros((N))
    for HQ,PR,DE in np.ndindex((N,N,N)):
        if PR != DE:
            EX[PR] += X[HQ,PR,DE]
            IM[DE] += X[HQ,PR,DE]
        if HQ != PR:
            OI[HQ] += X[HQ,PR,DE]
            II[PR] += X[HQ,PR,DE]
    return Zl,El,EX,IM,OI,II

# Simulate the model from the parameters and give output which is aggregated
def simulate_and_aggregate(N,theta,rho,tau_input,L,a,g,D):
    model = WWmodels.Model_single_industry(N=N,theta=theta,rho=rho,tau=tau_input,L=L,a=a,g=g,D=D)
    model.solve()
    Zl,El,EX,IM,OI,II = calc_aggregate(model.X,model.Z,model.E)
    return np.concatenate([Zl,El,EX,IM,OI,II])

def exacthatalgebra_and_aggregate(N,theta,rho,X,Z,E,D,tauhat_input):
    model = WWmodels.Model_single_industry(N=N,theta=theta,rho=rho,X=X,Z=Z,E=E,D=D)
    _,_,_,_,_,X1,Z1,E1 = model.exacthatalgebra(tauhat_input)
    Zl,El,EX,IM,OI,II = calc_aggregate(X1,Z1,E1)
    return np.concatenate([Zl,El,EX,IM,OI,II])

#%% Simulating scenario with solving a model
# 0. North and South
# 1. HFDI or VFDI 
# 2. trade liberalization / investment liberalization
# 3. Potential decomposition

# Set elasticities and market size
theta,rho = 4.5,0.55
N,L = 2,[1,1]

# Set a (So that South is more dirty)
a = np.ones((N,N,N))
a[1,:,:] = 1.1
# Set g (This only happens in trade)
g = np.ones((N,N,N))
for HQ,PR,DE in np.ndindex((N,N,N)):
    if PR == DE:
        g[HQ,PR,DE] = 0 
D = np.zeros((N))

# Calculate the step
HFDI_mplib,VFDI_mplib = np.empty((0,6*N)),np.empty((0,6*N))
HFDI_tradelib,VFDI_tradelib = np.empty((0,6*N)),np.empty((0,6*N))

trange = np.arange(0.75,1.25,0.05)
for t in trange:
    ## Investment liberalization
    tau_HFDI,tau_VFDI = set_tau(t,"mp")
    # Solve HFDI for mplib
    HFDI_mplib = np.vstack((HFDI_mplib,simulate_and_aggregate(N,theta,rho,tau_HFDI,L,a,g,D)))
    VFDI_mplib = np.vstack((VFDI_mplib,simulate_and_aggregate(N,theta,rho,tau_VFDI,L,a,g,D)))

    ## Trade liberalization
    tau_HFDI,tau_VFDI = set_tau(t,"trade")
    # Solve HFDI for tradelib
    HFDI_tradelib = np.vstack((HFDI_tradelib,simulate_and_aggregate(N,theta,rho,tau_HFDI,L,a,g,D)))
    VFDI_tradelib = np.vstack((VFDI_tradelib,simulate_and_aggregate(N,theta,rho,tau_VFDI,L,a,g,D)))

# Plot the figure for mp liberalization
column_names = ["Z_North","Z_South","E_North","E_South",
                "Export_North","Export_South","Import_North","Import_South",
                "OutwardMP_North","OutwardMP_South","InwardMP_North","InwardMP_South"]
df_HFDI = pd.DataFrame(data=HFDI_mplib,columns = column_names,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_mplib,columns = column_names,index=trange)

fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle("Investment liberalization", fontsize=15)
sns.set_style("whitegrid")
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,1])
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Export_North","OutwardMP_North"]],ax=axs[1,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Export_North","OutwardMP_North"]],ax=axs[1,1])
axs[0,0].set_title("HFDI")
axs[0,0].set_ylabel("emission")
axs[1,0].set_ylabel("economics")
axs[0,0].set_xlabel("tau")
axs[0,1].set_title("VFDI")
fig.savefig("./figures/mplib_simulation.png")

# Plot the figure for trade liberalization
df_HFDI = pd.DataFrame(data=HFDI_tradelib,columns = column_names,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_tradelib,columns = column_names,index=trange)
df_twocountry = pd.merge(left=df_HFDI,right=df_VFDI,how="inner",left_index=True,right_index=True)

# Plot the figure for trade liberalization
fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle("Trade liberalization", fontsize=15)
sns.set_style("whitegrid")
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,1])
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Export_North","OutwardMP_North"]],ax=axs[1,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Export_North","OutwardMP_North"]],ax=axs[1,1])
axs[0,0].set_title("HFDI")
axs[0,0].set_ylabel("emission")
axs[1,0].set_ylabel("economics")
axs[1,0].set_xlabel("tau")
axs[1,1].set_xlabel("tau")
axs[0,1].set_title("VFDI")
fig.savefig("./figures/trade_simulation.png")

#%% Simulating scenario using exact hat-algebra
# Set elasticities
theta,rho = 4.5,0.55
N = 2 

## Let's first set X 
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

# Sample allocation for symmetric country
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
Z_HFDI[1,:,:],Z_VFDI[1,:,:] = Z_HFDI[1,:,:] * 1.5,Z_VFDI[1,:,:] * 1.5
E_HFDI,E_VFDI = copy.deepcopy(X_HFDI),copy.deepcopy(X_VFDI)
for HQ,PR,DE in np.ndindex((N,N,N)):
    if PR == DE:
        E_HFDI[HQ,PR,DE],E_VFDI[HQ,PR,DE] = 0,0

# Calculate the step
HFDI_mplib,VFDI_mplib = np.empty((0,6*N)),np.empty((0,6*N))
HFDI_tradelib,VFDI_tradelib = np.empty((0,6*N)),np.empty((0,6*N))

trange = np.arange(0.75,1.25,0.05)
for t in trange:
    tauhat_HFDI,tauhat_VFDI = set_tauhat(t,"mp")
    HFDI_mplib = np.vstack((HFDI_mplib,exacthatalgebra_and_aggregate(N,theta,rho,X_HFDI,Z_HFDI,E_HFDI,D,tauhat_HFDI)))
    VFDI_mplib = np.vstack((VFDI_mplib,exacthatalgebra_and_aggregate(N,theta,rho,X_VFDI,Z_VFDI,E_VFDI,D,tauhat_VFDI)))
    tauhat_HFDI,tauhat_VFDI = set_tauhat(t,"trade")
    HFDI_tradelib = np.vstack((HFDI_tradelib,exacthatalgebra_and_aggregate(N,theta,rho,X_HFDI,Z_HFDI,E_HFDI,D,tauhat_HFDI)))
    VFDI_tradelib = np.vstack((VFDI_tradelib,exacthatalgebra_and_aggregate(N,theta,rho,X_VFDI,Z_VFDI,E_VFDI,D,tauhat_VFDI)))

# Set column names for tha figure
column_names = ["Z_North","Z_South","E_North","E_South",
                "Export_North","Export_South","Import_North","Import_South",
                "OutwardMP_North","OutwardMP_South","InwardMP_North","InwardMP_South"]

# Plot the figure for mp liberalization
df_HFDI = pd.DataFrame(data=HFDI_mplib,columns = column_names,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_mplib,columns = column_names,index=trange)
fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle("Investment liberalization", fontsize=15)
sns.set_style("whitegrid")
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,1])
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Export_North","OutwardMP_North"]],ax=axs[1,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Export_North","OutwardMP_North"]],ax=axs[1,1])
axs[0,0].set_title("HFDI")
axs[0,0].set_ylabel("emission")
axs[1,0].set_ylabel("economics")
axs[1,0].set_xlabel("tau")
axs[1,1].set_xlabel("tau")
axs[1,1].set_title("VFDI")
fig.savefig("./figures/mplib_simulation_eha.png")

# Plot the figure for trade liberalization
df_HFDI = pd.DataFrame(data=HFDI_tradelib,columns = column_names,index=trange)
df_VFDI = pd.DataFrame(data=VFDI_tradelib,columns = column_names,index=trange)
fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle("Trade liberalization", fontsize=15)
sns.set_style("whitegrid")
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Z_North","Z_South","E_North","E_South"]],ax=axs[0,1])
line_plot_HFDI = sns.lineplot(data=df_HFDI[["Export_North","OutwardMP_North"]],ax=axs[1,0])
line_plot_VFDI = sns.lineplot(data=df_VFDI[["Export_North","OutwardMP_North"]],ax=axs[1,1])
axs[0,0].set_title("HFDI")
axs[0,0].set_ylabel("emission")
axs[1,0].set_ylabel("economics")
axs[1,0].set_xlabel("tau")
axs[1,1].set_xlabel("tau")
axs[0,1].set_title("VFDI")
fig.savefig("./figures/trade_simulation_eha.png")
print("Done with exact hat-algebra")

