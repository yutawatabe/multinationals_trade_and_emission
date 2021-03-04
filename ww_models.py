# This module contains a class 
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

#%% Set a class for a WW model
class ModelSingleIndustry:
    """
    This class in a model in Wanner and Watabe (2021)
    where there is only a single indutry. Inputs are provided bin __init__ method.
    This class allows to 
        1. Solve a model and calculate equilibrium outcome (.solve)
        2. Back out deep parameters of the model from the equilibrium outcome (.inverse_solve)
        3. Calculate exact hat-algebra (.exacthatalgebra(tauhat))
    To check if the exact hat-algebra is valid, the verify command check if the 
    outcome of equilibrium recalculation coincides with that of exact hat-algebra.
    """

    def __init__(self, 
                N,   # Number of countries
                # Parameters and Settings
                theta = 4.5,  # Trade elasticity
                rho   = 0.55, # Multinationals elasticity                
                
                # Deep parameters
                tau = None, # Technology parameter
                L   = None, # Labor endowment
                a   = None, # Emission intensity
                g   = None, # Trade emission intensity
                D   = None, # Trade deficit

                # Equilibrium outcome
                Xm  = None, # Total expenditure of country m
                X   = None, # Allocation 
                w   = None, # Wage
                Z   = None, # Emission from production
                E   = None, # Emission from trade

                # Data (for Triangulation)
                # We also want wage data for this.
                T  = None, # Trade data
                M  = None, # Multinational production data
                Zl = None, # Observed emission for each country of production

                ):
                self.theta, self.rho, self.N, = theta, rho, N
                self.tau, self.L, self.a, self.g, self.D = tau, L, a, g, D
                self.Xm, self.X, self.w = Xm, X, w
                self.T,self.M,self.Zl = T, M, Zl
                self.Z,self.E = Z,E

                # Set data
                if X is not None:
                    # Calculate absorption, production and trade deficit
                    self.Xm = np.sum(X,axis=(0,1))
                    self.Ym = np.sum(X,axis=(0,2))
                    self.D  = self.Xm - self.Ym  
                elif T is not None:
                    # Calculate absorption, production and trade deficit
                    self.Xm = np.sum(T,axis=0)
                    self.Ym = np.sum(T,axis=1)
                    self.D  = self.Xm - self.Ym

                    # Normalize M to fit with T
                    # Adjust value in df_amne so that the total production in AMNE is equalized with 
                    # total production in WIODs. The ordering of the axis is different so be aware.
                    temp_M = copy.deepcopy(M)
                    for HQ,PR in np.ndindex(N,N):
                        temp_M[HQ,PR] = M[HQ,PR] * sum(T[PR,:]) / sum(M[:,PR])
                    M = copy.deepcopy(temp_M)
                    self.T, self.M = T, M


    def solve(self):
        """ 
        This function solves the equilibirum outcome (w,P,X,Z,E,u)
        from the paramters (tau,L,a,g) and the elasticity parameter theta and rho (which contained in the object).
        To solve the market clearing, this runs the fixed point iteration. Change the dumping parameter if there
        is some problem.
        """
        print("Start solving the model")
        # Solve the model from the parameter
        N = self.N
        # Set initial variable
        dif = 10
        w = np.ones(N)

        print("Searching over wages")
        while dif > 0.000001:
            # Set tentative value
            w_old = copy.deepcopy(w)
            p = np.empty((N,N,N))
            X  = np.zeros((N,N,N))
            Z  = np.zeros((N,N,N))
            E  = np.zeros((N,N,N))
            Xm = w * self.L + self.D
            Ym = w * self.L
    
            # Calculate price index and allocation
            for HQ,PR,DE in np.ndindex((N,N,N)):
                if np.isnan(self.tau[HQ,PR,DE]):
                    p[HQ,PR,DE] = np.nan
                else:
                    p[HQ,PR,DE] = w[PR] * self.tau[HQ,PR,DE]
            pi,_,Pm = self.calcpi(p)
            for HQ,PR,DE in np.ndindex((N,N,N)):
                X[HQ,PR,DE] = pi[HQ,PR,DE] * Xm[DE]

            # Check market clearing
            w = np.sum(X,axis=(0,2)) / self.L
            w = w/10 + w_old * 9/10
            w = w/w[0]
            dif = max(abs(w - w_old))
        print("Done searching over wages")

        # Calculate emission
        for HQ,PR,DE in np.ndindex((N,N,N)):
            # With scale effect 
            Z[HQ,PR,DE] = self.a[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * self.tau[HQ,PR,DE])
            E[HQ,PR,DE] = self.g[HQ,PR,DE] * X[HQ,PR,DE] / (w[PR] * self.tau[HQ,PR,DE])
            # Kill scale effect
            #Z[HQ,PR,DE] = self.a[HQ,PR,DE] * X[HQ,PR,DE] / w[PR] 
            #E[HQ,PR,DE] = self.g[HQ,PR,DE] * X[HQ,PR,DE] / w[PR]
        Z = np.nan_to_num(Z)
        E = np.nan_to_num(E)

        # Calculate utility
        ugoods = Xm / Pm
        uemission = 1 + (1/(np.sum(Z) + np.sum(E))**2)
        u = ugoods * uemission

        # Put eqm outcomes to the class
        print("Done solving eqm")
        self.u = u
        self.ugoods = ugoods
        self.uemission = uemission
        self.w = w
        self.X = X
        self.Z = Z
        self.E = E
        self.Xm = Xm
        self.Ym = Ym
    

    def exacthatalgebra(self,tauhat):
        """ 
        This method solves the equilibirum outcome in chnages
        from the eqm outcomes (X,Z,E) and the elasticity parameter theta and rho (which contained in the instance).
        This method requires tauhat, which is a ndarray((N,N,N)). This method does not use wage as an input.
        This method will return the alternative equilibrium outcome. This does not store any new variable in the class.
        """
        N = self.N
        
        # Setting boxes
        veps = self.theta / (1 - self.rho)
        X1   = np.zeros((N,N,N))

        # Calculate pi, pic, Pi
        pi = np.zeros((N,N,N))
        for HQ,PR,DE in np.ndindex((N,N,N)):
            pi[HQ,PR,DE] = self.X[HQ,PR,DE] / self.Xm[DE]
        pic = np.zeros((N,N,N))
        Pi = np.sum(pi,1)
        for HQ,PR,DE in np.ndindex((N,N,N)):
            if Pi[HQ,DE] == 0:
                pic[HQ,PR,DE] = 0
            else:
                pic[HQ,PR,DE] = pi[HQ,PR,DE] / Pi[HQ,DE]

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
                if pi[HQ,PR,DE] == 0:
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
                if pi[HQ,PR,DE] == 0:
                    pihat[HQ,PR,DE] == 0
                else:
                    pihat[HQ,PR,DE] = pihat_num[HQ,PR,DE] / (pihat_den1[HQ,DE] * pihat_den2[DE])

            # Update income
            Xm1 = self.Ym * what + self.D
            Ym1 = np.zeros((N))
            X1    = np.zeros((N,N,N))
            for HQ,PR,DE in np.ndindex((N,N,N)):
                Ym1[PR] += pi[HQ,PR,DE] * pihat[HQ,PR,DE] * Xm1[DE]
                X1[HQ,PR,DE] = pi[HQ,PR,DE] * pihat[HQ,PR,DE] * Xm1[DE] 
            what = Ym1 / self.Ym
            what = what * 1/10 + whatold * 9/10
            what = what / what[0]
            dif = max(abs(whatold - what))
            #print(dif) 

        # Update emission
        Z1 = np.zeros((N,N,N))
        E1 = np.zeros((N,N,N))
        for HQ,PR,DE in np.ndindex((N,N,N)):
            # Including scale effect
            Z1[HQ,PR,DE] = self.Z[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (phat[HQ,PR,DE] * self.Xm[DE])
            E1[HQ,PR,DE] = self.E[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (phat[HQ,PR,DE] * self.Xm[DE])
            # Kill scale effect
            #Z1[HQ,PR,DE] = self.Z[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (what[PR] * self.Xm[DE])
            #E1[HQ,PR,DE] = self.E[HQ,PR,DE] * (pihat[HQ,PR,DE] * Xm1[DE]) / (what[PR] * self.Xm[DE])
        return what/Pmhat,what,Pmhat,phat,pihat,X1,Z1,E1

    # Fill allocation from the data
    def fill_allocation(self,assumption):
        """
        This function fill in an allocation from the data.
        Assumptions are PFDI, HFDI, VFDI, RRC.
        For RRC, it will provide a parameter gamma and xi.
        """
        N,T,M = self.N,self.T,self.M
        X = np.zeros((N,N,N))
        self.assumption = assumption
        # Proportioanl export platform FDI
        if assumption == "PFDI":
            for HQ,PR,DE in np.ndindex(N,N,N):
                if M[HQ,PR] > 0:
                    X[HQ,PR,DE] = M[HQ,PR] / np.sum(M[:,PR]) * T[PR,DE]
            print("Proportional Export Platfrom FDI exists and set")
            self.X = X
        # Pure horizontal FDI
        elif assumption == "HFDI":
            for HQ,PR,DE in np.ndindex(N,N,N):
                if HQ == PR and PR == DE:
                    X[DE,DE,DE] = M[DE,DE] - sum(T[DE,:]) + T[DE,DE]
                elif PR == DE and HQ != DE:
                    X[HQ,DE,DE] = M[HQ,DE]
                elif HQ == PR and PR != DE:
                    X[HQ,HQ,DE] = T[HQ,DE]
                elif HQ == DE and PR != DE:
                    X[DE,PR,DE] = 0
            if np.any(X<0):
                print("Pure Horizontal FDI rejected")
            else:
                print("Pure Horizontal FDI exists and set")
                self.X = X
        # Pure vertical FDI
        elif assumption == "VFDI":
            for HQ,PR,DE in np.ndindex(N,N,N):
                if HQ == PR and PR == DE:
                    X[DE,DE,DE] = T[DE,DE]
                elif PR == DE and HQ != DE:
                    X[HQ,DE,DE] = 0
                elif HQ == PR and PR != DE:
                    X[HQ,HQ,DE] = T[HQ,DE] - M[DE,HQ]     
                elif HQ == DE and PR != DE:
                    X[DE,PR,DE] = M[DE,PR]      
            if np.any(X<0):
                print("Pure Vertical FDI rejected")
            else:
                print("Pure Vertical FDI exists and set")
                self.X = X
        # Ramondo Rodriguez-Clare FDI
        elif assumption == "RRC":
            # Warn that we need wage 
            if self.w is None:
                print("Wage does not exist")
            else:
                # Set default initial parameters
                gamma,xi = np.ones((N,N)),np.ones((N,N))
                count = 0
                dif = 1
                # Start searching over eqm gamma
                while dif > 0.001:
                    count += 1
                    gamma_old,xi_old = copy.deepcopy(gamma),copy.deepcopy(xi)
                    p = np.zeros((N,N,N))
                    for HQ,PR,DE in np.ndindex((N,N,N)):
                        p[HQ,PR,DE] = gamma[HQ,PR] * xi[PR,DE] * self.w[PR]
                    pi,_,_ = self.calcpi(p)
                    for HQ,PR,DE in np.ndindex((N,N,N)):
                        X[HQ,PR,DE] = pi[HQ,PR,DE] * self.Xm[DE]
                    M_model = np.sum(X,axis=2)
                    T_model = np.sum(X,axis=0)
                    if count % 2 == 0:
                        gamma = gamma * M_model / self.M * 1/10 + gamma_old * 9/10
                    else:
                        xi    = xi * T_model / self.T * 1/10 + xi_old * 9/10
                    # Normalize xi so that diagonal element is always 1
                    for DE in range(N):
                        xi[:,DE] = xi[:,DE] / xi[DE,DE]
                    dif = max(np.max(abs(self.T-T_model)),np.max(abs(self.M-M_model)))
                print("Ramondo Rodriguez-ClareFDI exists and set")
                self.X = X
                self.xi = xi
                self.gamma = gamma
        else:
            print("This assumption is not well defined")

    def fill_emission(self,emission_assumption):
        """
        This method will fill in the emission contents based
        on two assumptions: Only production loction matters
        and only headquarters location matters. WARNING is that
        this only works for RRC assumption.
        """
        N = self.N
        if self.assumption == "RRC":
            if self.w is None:
                print("Wage does not exist")
                return
            else:
                # Calculate quantity produced
                q = np.zeros((N,N,N))
                for HQ,PR,DE in np.ndindex(N,N,N):
                    q[HQ,PR,DE] = X[HQ,PR,DE] / (self.w[PR] * self.gamma[HQ,PR] * self.xi[PR,DE])
                if emission_assumption == "common_production_location":
                    # Calculate emission intensity and emission
                    ql = np.sum(q,axis=(0,2))
                    al = self.Zl / ql
                    self.Z = np.zeros((N,N,N))
                    for HQ,PR,DE in np.ndindex(N,N,N):
                        self.Z[HQ,PR,DE] = al[PR] * q[HQ,PR,DE]
                    print("Filled emission following common production location")
                elif emission_assumption == "common_headquarter_location":
                    # This is convenient to solve LP 
                    # Bit complex so verify if this is working
                    qli = np.sum(q,axis=2).T
                    ai = np.linalg.solve(qli,self.Zl)
                    if np.any(ai<0):
                        print("The common headquarter location emission intensity is rejected")
                    else:
                        print("The common headquarter location emission inteisty is not rejected")
                        self.Z = np.zeros((N,N,N))
                        for HQ,PR,DE in np.ndindex(N,N,N):
                            self.Z[HQ,PR,DE] = ai[HQ] * q[HQ,PR,DE]
                else:
                    print("The assumption is not currently in our plan")
        else:
            print("The assumption is not RRC.")        


    def verify_exacthatalgebra(self,tauhat):
        """
        This method checks if the result of exact hat-algebra
        matches the outcome that resolves the equilibrium.
        We need structural parameters to calculate this.
        """
        self.solve()
        tau0 = self.tau
        X0 = self.X
        w0 = self.w
        
        # First solve using exact hat-algebra
        _,what,_,_,_,X1_ha,Z1_ha,_ = self.exacthatalgebra(tauhat)
        w1_ha = what * w0

        # Resolve the eqm with new tau
        tau1 = tau0 * tauhat
        self.tau = tau1
        self.solve()
        X1_rs = self.X
        Z1_rs = self.Z
        w1_rs = self.w

        # Compare the outcome for resolving and exact hat-algebra
        X_compare = np.empty((N,N,N))
        Z_compare = np.empty((N,N,N))
        w_compare = w1_rs / w1_ha
        for HQ,PR,DE in np.ndindex((N,N,N)):
            if X0[HQ,PR,DE] ==0:
                X_compare[HQ,PR,DE] = 1
                Z_compare[HQ,PR,DE] = 1
            else:
                X_compare[HQ,PR,DE] = X1_rs[HQ,PR,DE] / X1_ha[HQ,PR,DE]
                Z_compare[HQ,PR,DE] = Z1_rs[HQ,PR,DE] / Z1_ha[HQ,PR,DE]
        print("Maximum difference (ratio) for the allocation is ")
        print(np.max(X_compare))
        print(np.min(X_compare))
        print("Maximum difference (ratio) for the production emission is ")
        print(np.max(Z_compare))
        print(np.min(Z_compare))        
        print("Maximum difference (ratio) for the wage is ")
        print(np.max(w_compare))
        print(np.min(w_compare))     
    

    def inverse_solve(self):
        """
        This method backs out structural parameters from the equilibrium outcome.
        We do need to know the deep parameters theta and rho.
        We need some explicit normalization on tau (in most general tau structure).
        """
        N = self.N

        # I think we need to know w
        w = self.w
        L = self.Xm / w

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
            for HQ,PR,DE in np.ndindex((N,N,N)):
                if self.X[HQ,PR,DE] == 0:
                    p[HQ,PR,DE] = np.nan
                else:
                    p[HQ,PR,DE] = p[HQ,PR,DE] * (X_model[HQ,PR,DE] / self.X[HQ,PR,DE])/10 + p_old[HQ,PR,DE] * 9/10
            p = p / p[0,0,0]
            dif = np.max(abs(X_model - self.X))     

        # Renormalize p (so that tau[DE,DE,DE] = 1)
        # This is not a trivial normalization. The result depends on this
        # way of normalization.
        for DE in range(N):
            p[:,:,DE] = p[:,:,DE] / p[DE,DE,DE] * w[DE]

        # Calculate tau, a and g
        tau = np.zeros((N,N,N))
        a   = np.zeros((N,N,N))
        g   = np.zeros((N,N,N))
        for HQ,PR,DE in np.ndindex((N,N,N)):
            tau[HQ,PR,DE] = p[HQ,PR,DE] / w[PR]
            a[HQ,PR,DE] = self.Z[HQ,PR,DE] * p[HQ,PR,DE] / self.X[HQ,PR,DE]
            g[HQ,PR,DE] = self.E[HQ,PR,DE] * p[HQ,PR,DE] / self.X[HQ,PR,DE]

        # Deep parameters
        self.tau = tau # Technology parameter
        self.L   = L # Labor endowment
        self.a   = a # Emission intensity
        self.g   = g # Trade emission intensity

    def calcpi(self,p):
        """ 
        This function calculate expenditure share from the price vector
        and the elasticity parameter theta and rho (which contained in the object)
        """
        N = self.N
        # Set tentative value
        Pim = np.zeros((N,N))
        Pm  = np.zeros(N)
        pi  = np.zeros((N,N,N))

        # Calculate price index and market share
        for HQ,PR,DE in np.ndindex((N,N,N)):
            if np.isnan(p[HQ,PR,DE]):
                Pim[HQ,DE] += 0
            else:
                Pim[HQ,DE] += p[HQ,PR,DE] ** (-self.theta/(1-self.rho))
        Pim = Pim ** (-(1-self.rho)/self.theta)
        for HQ,DE in np.ndindex((N,N)):
            Pm[DE] += Pim[HQ,DE] ** (-self.theta)
        Pm  = Pm ** (-1/self.theta)

        # Calculate expenditure share
        for HQ,PR,DE in np.ndindex((N,N,N)):
            if np.isnan(p[HQ,PR,DE]):
                pi[HQ,PR,DE] = 0
            else:
                pi[HQ,PR,DE] = (p[HQ,PR,DE] ** (-self.theta/(1-self.rho)) / (Pim[HQ,DE] ** (-self.theta/(1-self.rho)))
                              * Pim[HQ,DE] ** (-self.theta) / Pm[DE] ** (-self.theta)
                                )
        return pi,Pim,Pm

#%% Testing the class

if __name__ == '__main__' :
    print("""
            If run as main, it tests various examples
            print("First it solves two country equilibrium and exact hat-algebra
        """)
    # Set parameters
    theta = 4.5
    rho = 0.55
    
    # There are two countries North and South
    N = 2 
    # Set tau (but normalize things so that domestic tau will be tau)
    tau0 = np.ones((N,N,N))
    tau0[0,1,0] = np.nan
    L0 = [1,2]
    a0 = np.ones((N,N,N))
    g0 = np.ones((N,N,N))
    D0 = np.zeros((N))

    print("Set up a two country example and solve it")
    sample_model = ModelSingleIndustry(N=2,tau=tau0,L=L0,a=a0,g=g0,D=D0) 
    sample_model.solve()
    print(sample_model.X)

    print("Randomly changing the tau")
    tauhat0 = np.random.rand(N,N,N) + np.ones((N,N,N)) / 2

    print("Recover the deep parameter from the eqm outcome")
    print("Compare them and show they are close (or same)")
    print(sample_model.tau)
    sample_model.inverse_solve()
    print(sample_model.tau)

    print("Compare the exact hat-algebra and re-solving the model")
    sample_model.verify_exacthatalgebra(tauhat0)

    print("Second it triangulated three country example")
    #%% Triangulation
    # Simple sample case (where VFDI is a benchmark)
    # Symmetric 
    N = 3
    ai_temp = [1,2,1.5]
    g_temp = 1
    X = np.zeros((N,N,N))
    Z = np.zeros((N,N,N))
    E0 = np.zeros((N,N,N))
    for HQ,PR,DE in np.ndindex((N,N,N)):
        for HQ,PR,DE in np.ndindex(N,N,N):
            if HQ == PR and PR == DE:
                X[DE,DE,DE] = 3
            elif PR == DE and HQ != DE:
                X[HQ,DE,DE] = 0
            elif HQ == PR and PR != DE:
                X[HQ,HQ,DE] = 2     
            elif HQ == DE and PR != DE:
                X[DE,PR,DE] = 1       
            Z[HQ,PR,DE] = X[HQ,PR,DE] * ai_temp[HQ]
            E0[HQ,PR,DE] = X[HQ,PR,DE] * g_temp
    """
    N = 3
    X = np.random.rand(N,N,N) + 1
    """
    T0,M0,Zl0 = np.sum(X,axis=0),np.sum(X,axis=2),np.sum(Z,axis=(0,2))
    w0    = np.ones((N))
    model_triangulation = ModelSingleIndustry(N=N,theta=4.5,rho=0.55,T=T0,M=M0,w=w0,Zl=Zl0,E=E0)
    model_triangulation.fill_allocation("PFDI")
    print(model_triangulation.assumption)
    model_triangulation.fill_allocation("HFDI")
    print(model_triangulation.assumption)
    model_triangulation.fill_allocation("VFDI")
    print(model_triangulation.assumption)
    model_triangulation.fill_allocation("RRC")
    print(model_triangulation.assumption)
    model_triangulation.fill_emission("common_production_location")
    model_triangulation.fill_emission("common_headquarter_location")




