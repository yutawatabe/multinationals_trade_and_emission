# This script simulates two country model by Wanner and Watabe (2021)
import numpy as np
import copy

# Set parameters
theta = 4.5
rho = 0.55

# There are two countries North and South
N = 2 
# Set tau (but normalize things so that domestic tau will be tau)
tau = (np.random.rand(N,N,N) + 5) / 5
for DE in range(N):
    tau[:,:,DE] = tau[:,:,DE] / tau[DE,DE,DE] 
L = np.random.rand(N) + 1
a = np.random.rand(N,N,N) + 1
g = np.random.rand(N,N,N) + 1


#%% Nice functions

#%% Set a class for a mode
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
        while dif > 0.0001:
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
            pi,Pim,Pm = self.calcpi(p)
            for HQ,PR,DE in np.ndindex((N,N,N)):
                X[HQ,PR,DE] = pi[HQ,PR,DE] * Xm[DE]

            # Check market clearing
            w = np.sum(X,axis=(0,2)) / L
            w = w/10 + w_old * 9/10
            w = w/w[0]
            dif = max(abs(w - w_old))

        # Calculate emission
        for HQ,PR,DE in np.ndindex((N,N,N)):
            Z[HQ,PR,DE] = a[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * tau[HQ,PR,DE])
            E[HQ,PR,DE] = g[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * tau[HQ,PR,DE])
        
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
    
    def inverse_solve(self):
        # This problem back out the parameter from the observables
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

# We are able to recover the deep parameter from the eqm outcome 
# (but we need some explicit normalization)
testmodel = WWModel(N=2,tau=tau,L=L,a=a,g=g)
testmodel.solve()
testmodel.inverse_solve()

# We probably want to know what happens if RRC assumptions is made.
# Also we want to solve the hat-algebra