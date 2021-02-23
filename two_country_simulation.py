# This script simulates model by Wanner and Watabe (2021)
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

#%% Set a class for a WW model
class WWModel:

    def __init__(self, 
                N,   # Number of countries
                # Parameters and Settings
                theta = 4.5,  # Trade elasticity
                rho   = 0, # Multinationals elasticity                
                
                # Deep parameters
                tau = np.nan, # Technology parameter
                L   = np.nan, # Labor endowment
                a   = np.nan, # Emission intensity
                g   = np.nan, # Trade emission intensity

                # Equilibrium outcome
                Xm  = np.nan,  # Total expenditure of country m
                X   = np.nan,  # Allocation 
                w   = np.nan,  # Wage
                Z   = np.nan,  # Emission from production
                E   = np.nan,  # Emission from trade
                ):
                self.theta, self.rho, self.N, = theta, rho, N
                self.tau, self.L, self.a, self.g = tau, L, a, g
                self.Xm, self.X, self.w = Xm, X, w

                # Fill in pi (trade share)
                if not np.isnan(X):
                    pi = np.zeros((N,N,N))
                    for HQ,PR,DE in np.ndindex((N,N,N)):
                        print(pi)
                        pi[HQ,PR,DE] = X[HQ,PR,DE] / Xm[DE]
                else:
                    pi = np.nan
                self.pi = pi


    def calcpi(self,p):
        N = self.N
        # Set tentative value
        Pim = np.zeros((N,N))
        Pm  = np.zeros(N)
        pi  = np.zeros((N,N,N))

        # Calculate price index and market share
        for HQ,PR,DE in np.ndindex((N,N,N)):
            Pim[HQ,DE] += p[HQ,PR,DE] ** (-self.theta/(1-self.rho))
        Pim = Pim ** (-(1-self.rho)/self.theta)
        for HQ,DE in np.ndindex((N,N)):
            Pm[DE] += Pim[HQ,DE] ** (-self.theta)
        Pm  = Pm ** (-1/self.theta)

        # Calculate expenditure share
        for HQ,PR,DE in np.ndindex((N,N,N)):
            pi[HQ,PR,DE] = (p[HQ,PR,DE] ** (-self.theta/(1-self.rho)) / (Pim[HQ,DE] ** (-self.theta/(1-self.rho)))
                          * Pim[HQ,DE] ** (-self.theta) / Pm[DE] ** (-self.theta)
                        )
        return pi,Pim,Pm


    def solve(self):
        # Solve the model from the parameter
        N = self.N
        # Set initial variable
        dif = 10
        w = np.ones(N)

        # Start solving 
        while dif > 0.000001:
            # Set tentative value
            w_old = copy.deepcopy(w)
            p = np.zeros((N,N,N))
            X  = np.zeros((N,N,N))
            Z  = np.zeros((N,N,N))
            E  = np.zeros((N,N,N))
            Xm = w * L
    
            # Calculate price index and allocation
            for HQ,PR,DE in np.ndindex((N,N,N)):
                p[HQ,PR,DE] = w[PR] * self.tau[HQ,PR,DE]
            pi,_,Pm = self.calcpi(p)
            for HQ,PR,DE in np.ndindex((N,N,N)):
                X[HQ,PR,DE] = pi[HQ,PR,DE] * Xm[DE]

            # Check market clearing
            w = np.sum(X,axis=(0,2)) / L
            w = w/10 + w_old * 9/10
            w = w/w[0]
            dif = max(abs(w - w_old))

        # Calculate emission
        for HQ,PR,DE in np.ndindex((N,N,N)):
            Z[HQ,PR,DE] = a[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * self.tau[HQ,PR,DE])
            E[HQ,PR,DE] = g[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * self.tau[HQ,PR,DE])
        
        # Calculate utility
        ugoods = w / Pm
        uemission = 1 + (1/(np.sum(Z) + np.sum(E))**2)
        u = ugoods * uemission

        # Put eqm outcomes to the class
        self.u = u
        self.ugoods = ugoods
        self.uemission = uemission
        self.w = w
        self.X = X
        self.pi = pi
        self.Z = Z
        self.E = E
        self.Xm = Xm
    

    def exacthatalgebra(self,tauhat):
        # This method calculates exact hat-algebra 
        # given tauhat (change in tau) in terms of ratio
        N = self.N
        
        # Setting boxes
        veps = self.theta / (1 - self.rho)
        X1   = np.zeros((N,N,N))
        Ym = np.sum(self.X,axis=(0,2))
        #D    = X_m - Y_m
    
        # Calculate pi, pic, Pi
        pic = np.zeros((N,N,N))
        Pi = np.sum(self.pi,1)
        for HQ,PR,DE in np.ndindex((N,N,N)):
            if Pi[HQ,DE] == 0:
                pic[HQ,PR,DE] = 0
            else:
                pic[HQ,PR,DE] = self.pi[HQ,PR,DE] / Pi[HQ,DE]

        # Put an initial guess on what and phat
        whatold = np.ones(N) * 0.95
        what = np.ones(N)
        dif = 1
        while  dif> 0.000001:
            # Save the current wage_hat
            whatold = copy.deepcopy(what)

            # Initialize price index hat and pi hat  
            phat  = np.zeros((N,N,N))
            Pimhat = np.zeros((N,N))
            Pmhat = np.zeros(N)
            pihat_num = np.zeros((N,N,N))
            pihat_den1 = np.zeros((N,N))
            pihat_den2 = np.zeros((N))
            pihat     = np.zeros((N,N,N))

            # Calculate price index change
            for HQ,PR,DE in np.ndindex((N,N,N)):
                phat[HQ,PR,DE] = what[PR] * tauhat[HQ,PR,DE]
                Pimhat[HQ,DE] += pic[HQ,PR,DE] * phat[HQ,PR,DE]**(-veps)
            for HQ,DE in np.ndindex((N,N)):
                if Pi[HQ,DE] == 0:
                    Pimhat[HQ,DE] = 0
                else:
                    Pimhat[HQ,DE] = Pimhat[HQ,DE] ** (-1/veps)
            for HQ,DE in np.ndindex((N,N)):
                if Pi[HQ,DE] == 0:
                    Pmhat[DE] += 0
                else:
                    Pmhat[DE] += Pi[HQ,DE] * Pimhat[HQ,DE] ** (-self.theta)
            Pmhat = Pmhat ** (-1/self.theta)
            # Calculate pihat and pi1
            for HQ,PR,DE in np.ndindex((N,N,N)):
                if self.pi[HQ,PR,DE] == 0:
                    pihat_num[HQ,PR,DE] = 0
                    pihat_den1[HQ,DE] += 0
                else:
                    pihat_num[HQ,PR,DE] = Pimhat[HQ,DE]**(-self.theta) * phat[HQ,PR,DE] **(-veps)
                    pihat_den1[HQ,DE] += pic[HQ,PR,DE] * phat[HQ,PR,DE]**(-veps)
            for HQ,DE in np.ndindex((N,N)):
                if Pi[HQ,DE] == 0:
                    pihat_den2[DE] += 0
                else:
                    pihat_den2[DE] += Pi[HQ,DE] * Pimhat[HQ,DE]**(-self.theta)
            for HQ,PR,DE in np.ndindex((N,N,N)):
                if self.pi[HQ,PR,DE] == 0:
                    pihat[HQ,PR,DE] == 0
                else:
                    pihat[HQ,PR,DE] = pihat_num[HQ,PR,DE] / (pihat_den1[HQ,DE] * pihat_den2[DE])

            # Update income
            Xm1 = Ym * what
            Ym1 = np.zeros((N))
            X1    = np.zeros((N,N,N))
            for HQ,PR,DE in np.ndindex((N,N,N)):
                Ym1[PR] += self.pi[HQ,PR,DE] * pihat[HQ,PR,DE] * Xm1[DE]
                X1[HQ,PR,DE] = self.pi[HQ,PR,DE] * pihat[HQ,PR,DE] * Xm1[DE] 
            what = Ym1 / Ym
            what = what * 1/8 + whatold * 7/8
            what = what / what[0]
            dif = max(abs(whatold - what))
    
        # Update emission
        Z1 = np.zeros((N,N,N))
        E1 = np.zeros((N,N,N))
        for HQ,PR,DE in np.ndindex((N,N,N)):
            Z1[HQ,PR,DE] = self.Z[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (phat[HQ,PR,DE] * self.Xm[DE])
            E1[HQ,PR,DE] = self.E[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (phat[HQ,PR,DE] * self.Xm[DE])
        
        #return wage_hat
        return what/Pmhat,what,Pmhat,phat,pihat,X1,Z1,E1

    def verify_exacthatalgebra(self,tauhat):
        # This method checks if the result of exact hat-algebra
        # matches the outcome that resolves the eqm
        # We need deep parameter (although we can back it out from eqm out)
        self.solve()
        tau0 = self.tau
        X0 = self.X
        Z0 = self.Z
        w0 = self.w
        
        # First solve using exact hat-algebra
        _,what,_,_,_,X1_ha,Z1_ha,E1_ha = self.exacthatalgebra(tauhat)
        w1_ha = what * w0

        # Resolve the eqm with new tau
        tau1 = tau0 * tauhat
        self.tau = tau1
        self.solve()
        X1_rs = self.X
        Z1_rs = self.Z
        w1_rs = self.w

        # Compare
        X_compare = X1_rs / X1_ha
        Z_compare = Z1_rs / Z1_ha
        w_compare = w1_rs / w1_ha
        print(X_compare)
        print(Z_compare)
        print(w_compare)

    def inverse_solve(self):
        # This method back out the parameter from the observables
        # Maybe we don't need this because we only look at exact hat-algebra
        # but we will see...
        N = self.N

        # I think we need to know w
        w = self.w

        # Start backing up price (with some normalization)
        p = np.ones((N,N,N))
        X_model = np.zeros((N,N,N))
        dif = 10

        while dif > 0.001:
            # Calculate pi and X from the parameter p
            pi_model,_,_ = self.calcpi(p)
            for HQ,PR,DE in np.ndindex((N,N,N)):
                X_model[HQ,PR,DE] = pi_model[HQ,PR,DE] * self.Xm[DE] 

            # Update p
            p_old = copy.deepcopy(p)
            p = p * (X_model / self.X)/10 + p_old * 9/10
            p = p / p[0,0,0]
            dif = np.max(abs(X_model - self.X))     

        # Renormalize p (so that tau[DE,DE,DE] = 1)
        # This is not a trivial normalization. The result depends on this
        # way of normalization.
        for DE in range(N):
            p[:,:,DE] = p[:,:,DE] / p[DE,DE,DE] * w[DE]

        # Calculate tau, a and g
        tau = np.zeros((N,N,N))
        for HQ,PR,DE in np.ndindex((N,N,N)):
            tau[HQ,PR,DE] = p[HQ,PR,DE] / w[PR]
            a[HQ,PR,DE] = self.Z[HQ,PR,DE] * p[HQ,PR,DE] / self.X[HQ,PR,DE]
            g[HQ,PR,DE] = self.E[HQ,PR,DE] * p[HQ,PR,DE] / self.X[HQ,PR,DE]

        # Deep parameters
        self.tau = tau # Technology parameter
        self.L   = L # Labor endowment
        self.a   = a # Emission intensity
        self.g   = g # Trade emission intensity

#%% Testing the class

# Set parameters
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 
# Set tau (but normalize things so that domestic tau will be tau)
tau = (np.random.rand(N,N,N) + 5) / 5
for DE in range(N):
    tau[:,:,DE] = tau[:,:,DE] / tau[DE,DE,DE] 
tau = np.ones((N,N,N))
L = np.random.rand(N) + 1
a = np.random.rand(N,N,N) + 1
g = np.random.rand(N,N,N) + 1
tauhat = np.ones((N,N,N))
for HQ,PR,DE in np.ndindex((N,N,N)):
    if HQ != PR:
        tauhat[HQ,PR,DE] = 1.10


# We are able to recover the deep parameter from the eqm outcome 
# (but we need some explicit normalization)
testmodel = WWModel(N=2,tau=tau,L=L,a=a,g=g)
testmodel.solve()
testmodel.inverse_solve()
# We probably want to know what happens if RRC assumptions is made.
# Also we want to solve the hat-algebra

# Construct tauhat and calculate exact hat-algebra
hat_result = testmodel.exacthatalgebra(tauhat)
# Cmpare the exact hat-algebra and re-solving the model
# That is done. Good.
testmodel.verify_exacthatalgebra(tauhat)
print("end")

#%% Simulating scenario
# 0. North and South
# 1. HFDI or VFDI 
# 2. trade liberalization / investment liberalization
# 3. Potential decomposition