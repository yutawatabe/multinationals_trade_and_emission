# -*- coding: utf-8 -*-
"""
Created on 2021/03/04
This module import a class NeosAmpl that sends AMPL jobs to NEOS
@author: Yuta Watbe
"""

import os
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
import xmlrpc.client as xmlrpclib
from itertools import product

import numpy as np
import pandas as pd
import ww_models

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
                 modfile="",     # A name of a template of the mod file that describe the problem (xml file)
                 datfile="",     # Name of the txt file that include dat description
                 # Information of the solver
                 category="",        # category of the problem
                 solver="",          # what solver to use
                 solver_option="",   # solver option
                 # For note
                 settings = {}, # Settings that we want to include as information
                 ):       

        # Set attributes
        self.parameters,self.min_max,self.obj_func = parameters,min_max,obj_func
        self.modfile,self.datfile = modfile,datfile
        self.category,self.solver,self.solver_option = category,solver,solver_option
        self.settings = settings
    
        for attr_name,attr_value in (parameters|settings).items():
            setattr(self,attr_name,attr_value)

        # Switch minimize and maximize if objective function has reciprocal
        if re.search("RECIPROCAL",self.obj_func):
            if min_max == "minimize":
                min_max = "maximize"
            elif min_max == "maximize":
                min_max = "minimize"

        # Start parsing xml file and input information
        # xml goes as [category,solver,inputMethod,priority,email,model,data,command,comment]
        xml = ET.parse(modfile).getroot()
        xml[0].text = category
        xml[1].text = solver
        xml[6].text = ET.parse(datfile).getroot().text
        
        # Replacing command text (7th of xml file)
        xml[7].text = (xml[7].text.replace('objective',min_max + " " + "obj_output" + " " + ":" + obj_func)
                                  .replace('solver_option','option' + ' ' + solver_option + " "))    
        # Add comment so it's easy to check in email
        xml[8].text = (xml[8].text + min_max + " " + obj_func + " " + " using " + solver)

        # Set parameters to model template and comments 
        for param_name,param_value in parameters.items():
            param_temp = param_name + "_val"
            xml[7].text = xml[7].text.replace(param_temp,str(param_value))
            xml[8].text += param_name + ":" + str(param_value)

        self.xml = xml
        self.value = np.nan
        self.ub    = np.nan
        self.lb    = np.nan
        self.status = "unsubmitted"
        self.solve_result = ""

    def add_to_model(self,text):
        """
        This method adds text to the model section of xml
        """
        # Start parsing xml file and input information
        # xml goes as [category,solver,inputMethod,priority,email,data,command,comment]
        self.xml[5].text = self.xml[5].text + text 

    def submit(self):
        # Submit job to server
        # Set a server
        SERVER = xmlrpclib.Server("https://neos-server.org:3333")  
        xml_text = ET.tostring(self.xml).decode()
        self.jobNumber,self.jobPassword = SERVER.submitJob(xml_text)
        self.status = "submitted"

    def kill(self):
        # Kill jobs
        SERVER.killJob(self.jobNumber,self.jobPassword)
        self.status = "killed"

    def check_done(self):
        # Set a server
        SERVER = xmlrpclib.Server("https://neos-server.org:3333")  
        try:
            status = SERVER.getJobStatus(self.jobNumber,self.jobPassword)
            return status != "Running"
        except:
            print("There's something wrong with the job status")
            print(self.jobNumber)
            print(self.jobPassword)
            return None

    def retrieve(self):
        if self.status == "submitted":
            # Set a server
            SERVER = xmlrpclib.Server("https://neos-server.org:3333")  
            result = SERVER.getFinalResults(self.jobNumber,self.jobPassword).data.decode()
            self.result = result
            obj_line     = re.search("obj_output" + " = (.*)\n", result)
            obj_ub_line  = re.search("upper bound" + " = (.*)\n" , result)
            obj_lb_line  = re.search("Objective lower bound" + " = (.*),", result)
            solve_result = re.search("solve_result" + " = (.*)\n", result)
            memory_limit = re.search("Error: Your job was terminated because it exceeded the maximum allotted memory for a job.",result)
            if obj_line:
                self.value = float(obj_line.group(1))
                self.status = "successful"
            else:
                self.status = "failed"
                return
            if solve_result:
                self.solve_result = solve_result.group(1)       
            if obj_ub_line:
                self.ub = float(obj_ub_line.group(1))
            if obj_lb_line:
                self.lb = float(obj_lb_line.group(1))     
            # Take reciprocal if objective function takes reciprocal of original
            if re.search("RECIPROCAL",self.obj_func):
                self.value = 1/self.value
                if not np.isnan(self.lb) and self.lb != 0 and self.ub != 0:
                    self.ub, self.lb  = 1/self.lb, 1/self.ub
            # Power by 1 / theta is there is no theta 
            if re.search("NOTHETA",self.obj_func):
                self.value = self.value ** (1/self.theta)
                if not np.isnan(self.lb) and self.lb != 0 and self.ub != 0:
                    self.ub, self.lb  = self.ub **(1/self.theta), self.lb ** (1/self.theta)
        elif self.status == "unsubmitted":
            print("The problem is not submitted yet")
        elif self.status == "killed":
            print("The job is killed")
        return

    def retrieve_variable(self,variable):
        """
        This method retrieves variable from the NEOS result.
        The input variables is a string object, and the output is variables in the string
        and the float.
        """
        # The input variable is a string object
        var_line = re.search(variable + " :="+ ".*?" + ";",self.result,re.DOTALL)
        var_line_table = re.search(variable + " \[" + ".*?" + ";",self.result,re.DOTALL)
        if var_line:
            var_text      = var_line.group(0)
            temp = re.compile(r'\d+(?:\.\d*)') 
            variable = [float(element) for element in var_line.group(0).split() if temp.match(element)]        
            return var_text,variable
        elif var_line_table:
            var_text      = var_line_table.group(0)
            temp = re.compile(r'\d+(?:\.\d*)') 
            variable = [float(element) for element in var_line_table.group(0).split() if temp.match(element)]                  
            return var_text,variable
        else:
            print("There is no variables displayed")
            return None,None

    def to_DataFrame(self):
        """
        This method exports Neos into DataFrame style.
        """
        df_Neos = pd.DataFrame(
                {'parameters': str(self.parameters),
                'min_max':self.min_max,
                'obj_func':self.obj_func,
                'value':self.value,
                'lb':self.lb,
                'ub':self.ub,
                'modfil':self.modfile,
                'datfile':self.datfile,
                'jobNumber':str(self.jobNumber),
                'jobPassword':str(self.jobPassword),
                'solver':self.solver,   
                'solver_option':self.solver_option,
                'solve_result':self.solve_result,
                'status':self.status,
                'settings':str(self.settings),
                },
                    index=[self.jobNumber]
                    )
        return df_Neos

def submit_and_retrieve_NEOSs(NEOSs,file): 
    """
    Retreive and resubmit results. For the list of problems (Neos instances)
    it enumerates all the problems to check the condition. 
    """
    problems = list(NEOSs.keys())
    jobs_limit = 15
    solved = 0
    ongoing = 0
    while solved < len(NEOSs):
        ongoing = max([ongoing,0])
        print(solved)
        print(ongoing)
        problem = problems.pop(0)      
        print(problem)
        # Submit job if it is not submitted yet 
        # and has less jobs than 15 jobs
        if (NEOSs[problem].status == "unsubmitted") & (ongoing < jobs_limit):
            print("Unbsumitted. Trying to submit")
            try:
                NEOSs[problem].submit()
                time.sleep(30)
                ongoing += 1
            except:
                print("Submission failed. Skip and pass to the next problem")
                print(sys.exc_info())
            problems.append(problem)
        # Do not submit job if there's too many jobs running
        elif (NEOSs[problem].status == "unsubmitted") & (ongoing >= jobs_limit):
            print("Too many jobs running")
            problems.append(problem)
        # Retrive job if it was submitted
        elif NEOSs[problem].status == "submitted":
                if NEOSs[problem].check_done():
                    print("Done, retrieve")
                    NEOSs[problem].retrieve()
                    ongoing -= 1
                else:
                    print("Not done yet")
                    problems.append(problem)
        
        # Check the result
        if NEOSs[problem].status == "failed":
            print("There's something wrong print the result")
            print(NEOSs[problem].jobNumber)
            print(NEOSs[problem].jobPassword)
            try:
                problem.submit()
                ongoing += 1
                time.sleep(30)
                NEOSs[problem].status == "submitted"
            except:
                print("Retried, but submission failed")
                print(sys.exc_info())
            problems.append(problem)
        elif NEOSs[problem].status == "successful":
            solved += 1
        with open(file,'wb') as f1:
            pickle.dump(NEOSs,f1)  
        time.sleep(10)  

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
    X = np.random.rand(N,N,N) + 1
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
            # Calculate emission
            Z[HQ,PR,DE] = X[HQ,PR,DE] * ai[HQ]
            E0[HQ,PR,DE] = X[HQ,PR,DE] * g
    T0,M0,Zl0 = np.sum(X,axis=0),np.sum(X,axis=2),np.sum(Z,axis=(0,2))
    w0 = np.ones(N)

    #%% Calculate using exact hat-algebra
    # Create model object to calculate things
    test_model = ww_models.ModelSingleIndustry(N,theta=theta_benchmark,
                                                 rho=rho_benchmark,
                                                 w=w0,E=E0,T=T0,M=M0,Zl=Zl0)
    test_model.fill_allocation("RRC")
    test_model.fill_emission("common_production_location")

    # Include tauhat and calculate hat-algebra with the assumption
    tauhat = np.random.rand(N,N,N) / 5  + 1 
    rwhat,what,Pmhat,phat,pihat,X1,Z1,E1 = test_model.exacthatalgebra(tauhat)
    TotalEmission = np.sum(Z1) + np.sum(E1)

    #%% Start calculating exact hat-algebra using NEOS AMPL Knitro
    # Set data into dataframe
    df_E = pd.DataFrame(columns=["E"])
    df_tauhat = pd.DataFrame(columns=["tauhat"])
    for HQ,couHQ in enumerate(countries):
        for PR,couPR in enumerate(countries):
            for DE,couDE in enumerate(countries):
                df_E.loc[couHQ + " " + couPR + " " + couDE] = [E0[HQ,PR,DE]]
                df_tauhat.loc[couHQ + " " + couPR + " " + couDE] = [tauhat[HQ,PR,DE]]
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
    data.write("\n")
    data.write("param tauhat := \n " + df_tauhat.to_string(header=False,index_names=False) + ';')
    data.write("]]></data>")
    data.close()

    # Create NEOS object
    test_NEOS = NEOS(parameters={'theta':theta_benchmark,
                                 'rho':rho_benchmark},
                    min_max="minimize",
                    obj_func="TotalEmission",
                    modfile="./xmls/RRC_emission.xml",
                    datfile="./tempfolder/data.xml",
                    category="cp",
                    solver="Knitro",
                    solver_option='knitro_options "ms_enable=1 ms_maxsolves=10 ms_maxtime_real=25200 outlev=2";',
                    settings={"explanation":"sample project with three countries"})
    test_NEOS.add_to_model("subject to CommonProductionLocation {(HQ0,HQ1,PR,DE0,DE1) in FIVELAT}: a[HQ0,PR,DE0]= a[HQ1,PR,DE1];")
    #test_NEOS.add_to_model("subject to CommonHeadquartersLocation {(HQ,PR0,PR1,DE0,DE1) in FIVELAT}: a[HQ,PR0,DE0]= a[HQ,PR1,DE1];")

    test_NEOS.submit()
    test_NEOS.retrieve()
    print(test_NEOS.to_DataFrame())
    print("Check the result")

    
    
