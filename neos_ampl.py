# -*- coding: utf-8 -*-
"""
Created on 2021/03/04
This module import a class NeosAmpl that sends AMPL jobs to NEOS
@author: Yuta Watbe
"""

import re
import xmlrpc.client as xmlrpclib
import xml.etree.ElementTree as ET
import time
import sys
import pandas as pd
import pickle
import os
from itertools import product
import numpy as np

if __name__ == "__main__":
    print("This is a module that import NEOS_problem. Import to use.")

# Set server
SERVER = xmlrpclib.Server("https://neos-server.org:3333")  

#%% Set class NEOS
class NEOS:
    def __init__(self,
                 # Information of the problem
                 parameters= {}, # Dictionary of parameter values to be filled in
                 min_max ="",    # Whether we want to minimize or maximize
                 obj_func="",    # Objective function we want to minimize
                 model="",       # A template of mod file that describe the problem (xml file)
                 dat="",         # Name of the txt file that include dat description
                 # Information of the solver
                 category="",        # category of the problem
                 solver="",          # what solver to use
                 solver_option="",   # solver option
                 # For note
                 settings = {}, # Settings that we want to include as information
                 ):       

        # Set attributes
        self.min_max,self.obj_func = min_max,obj_func
        self.category,self.solver,self.solver_option = category,solver,solver_option
        for attr_name,attr_value in (parameters|settings).items():
            setattr(self,attr_name,attr_value)

        # Switch minimize and maximize if objective function has reciprocal
        if re.search("RECIPROCAL",self.obj_func):
            if min_max == "minimize":
                min_max = "maximize"
            elif min_max == "maximize":
                min_max = "minimize"

        # Start parsing xml file and input information
        # xml goes as [category,solver,inputMethod,priority,email,data,command,comment]
        xml = ET.parse(model).getroot()
        xml[0].text = category
        xml[1].text = solver
        xml[6].text = ET.parse(dat).getroot().text
        
        # Replacing command text (7th of xml file)
        xml[7] = (xml[7].replace('objective_function',min_max + " " + "obj_output" + " " + ":" + obj_func)
                        .replace('solver_option','option' + ' ' + solver_option + " "))    
        # Add comment so it's easy to check in email
        xml[8].text = (xml[8].text + min_max + " " + obj_func + " " + " using " + solver)

        # Set parameters to model template and comments
        for param_name,param_value in parameters.items():
            param_temp = param_name + "_val"
            xml[7].text = xml[7].text.replace(param_temp,str(param_value))
            xml[8].text += param_name + ":" + str(param_value)

        self.xml_text = ET.tostring(xml).decode()
        self.value = np.nan
        self.ub    = np.nan
        self.lb    = np.nan
        self.status = "unsubmitted"
        self.solve_result = ""

#%% Simulate three country model and fill solve the RRC
if __name__ == '__main__' :
    print("""
            If run as main, it tests simple two country examples
        """)

    # Settings
    N,countries = 3,["DEU","JPN","USA"]
    theta_benchmark = 4.5
    rho_benchmark   = 0.55

    # Generate allocation and data
    ai,g = [1,2,1.5],1
    X,Z,E0 = np.zeros((N,N,N)),np.zeros((N,N,N)),np.zeros((N,N,N))
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
            Z[HQ,PR,DE] = X[HQ,PR,DE] * ai[HQ]
            E0[HQ,PR,DE] = X[HQ,PR,DE] * g
    T0,M0,Zl0 = np.sum(X,axis=0),np.sum(X,axis=2),np.sum(Z,axis=(0,2))
    w0 = np.ones(N)

    # Set data into dataframe
    df_E = pd.DataFrame(columns=["E"])
    for HQ,couHQ in enumerate(countries):
        for PR,couPR in enumerate(countries):
            for DE,couDE in enumerate(countries):
                df_E.loc[couHQ + " " + couPR + " " + couDE] = [E0[HQ,PR,DE]]
    df_M = pd.DataFrame(data=M0,index=countries,columns=countries)
    df_T = pd.DataFrame(data=T0,index=countries,columns=countries)
    df_w = pd.DataFrame(data=w0,index=countries,columns=["w"])
    df_Zl = pd.DataFrame(data=Zl0,index=countries,columns=["Zl"])
                
    #% Translate data to XML file for NEOS server
    data = open("tempfolder/data.xml","w") 
    data.write("<data><![CDATA[")
    data.write("set COU := " + " ".join(countries) + ";")
    data.write("\n")
    data.write("param w := \n " + df_w.to_string(header=False,index_names=False) + ';')
    data.write("\n")
    data.write("param Zl := \n " + df_Zl.to_string(header=False,index_names=False) + ';')
    data.write("\n")
    data.write("param E := \n " + df_E.to_string(header=False,index_names=False) + ';')
    data.write("\n")
    data.write( "param mprod : " + " ".join(list(countries)) + " := \n" 
                              + df_M.to_string(header=False,index_names=False) + ';') 
    data.write("\n")
    data.write( "param trade : " + " ".join(list(countries)) + " := \n" 
                              + df_T.to_string(header=False,index_names=False) + ';')
    data.write("]]></data>")
    data.close()





    
    
