# This script simulates two country model by Wanner and Watabe (2021)
import numpy as np
import copy

# Set parameters
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 
tau = (np.random.rand(N,N,N) + 10) / 10
#tau = np.ones((N,N,N))
L = np.ones(N)
a = np.random.rand(N,N,N) + 1
g = np.random.rand(N,N,N) + 1

#%% Solve an equilibrium

# Set initial wage
w = np.ones(N)

# Start solving 
for _ in range(50):
    # Set tentative value
    w_old = copy.deepcopy(w)
    p = np.zeros((N,N,N))
    Pim = np.zeros((N,N))
    Pm  = np.zeros(N)
    X  = np.zeros((N,N,N))
    Xm = w * L
    
    # Calculate price index
    for HQ,PR,DE in np.ndindex((N,N,N)):
        p[HQ,PR,DE] = w[PR] * tau[HQ,PR,DE]
    for HQ,PR,DE in np.ndindex((N,N,N)):
        Pim[HQ,DE] += p[HQ,PR,DE] ** (-theta/(1-rho))
    Pim = Pim ** (-(1-rho)/theta)
    for HQ,DE in np.ndindex((N,N)):
        Pm[DE] += Pim[HQ,DE] ** (-theta)
    Pm  = Pm ** (-1/theta)

    # Calculate expenditure
    for HQ,PR,DE in np.ndindex((N,N,N)):
        X[HQ,PR,DE] = (p[HQ,PR,DE] ** (-theta/(1-rho)) / (Pim[HQ,DE] ** (-theta/(1-rho)))
                      * Pim[HQ,DE] ** (-theta) / Pm[DE] ** (-theta)
                      ) * Xm[DE]

    # Check market clearing
    w = np.sum(X,axis=(0,2)) / L
    w = w/10 + w_old * 9/10
    w = w/w[0]
    dif = w - w_old
    print(w)

print("End")


