"""
    This file runs main script for the paper
"""

import copy
import os
import pickle
import xmlrpc.client as xmlrpclib
import numpy as np
import pandas as pd
import ww_models
from itertools import product
from logging import DEBUG, WARNING, FileHandler, StreamHandler, Formatter, getLogger

def main_test():

    SERVER = xmlrpclib.Server("https://neos-server.org:3333")  

    # Create a folder
    result_dir = "testfolder/main"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Set logger
    logger = getLogger("main")
    logger.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(DEBUG)
    fh = FileHandler(result_dir + '/main.log')
    fh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("The results are stored in {0}, and the log is stored as {1}".format(result_dir,"main.log"))
    logger.info("This is a test script for io linkage model")

    #%% Set elasticity of subsitution
    theta_benchmark = 4.5
    rho_benchmark   = 0.55

    #%% Simualte and show two country model
    two_country_example(result_dir,theta_benchmark,rho_benchmark,logger)

    #%% Simulate three country model
    N = 3
    set_use = ["DEU","JPN","USA"]
    set_use = sorted(set_use)
    logger.info("""I use parameter as follows:
    The benchmark theta is {0}. The benchmark rho is {1}.
    I use {2} countries: {3}""".format(theta_benchmark,rho_benchmark,N,' '.join(set_use)))

    # Clean the test data
    clean_testdata(result_dir,set_use,theta_benchmark,rho_benchmark,N,logger)

    # Calculate GO for the special case
    calc_GOGTGM_SpecialCases(result_dir,set_use,theta_benchmark,rho_benchmark,logger)

    # Calculate GO for the bounds (is it possible? I don't know)
    calc_GO(result_dir,set_use,theta_benchmark,rho_benchmark,logger)

    # Calculate various exact hat algebra results for the specialcase
    calc_ExactHatAlgebra_SpecialCases(result_dir,set_use,theta_benchmark,rho_benchmark,logger)

    # Calculate various exact hat algebra results for bound
    calc_ExactHatAlgebra(result_dir,set_use,theta_benchmark,rho_benchmark,logger)

def clean_testdata(result_dir,set_use,theta,rho,N,logger):

    truemodel = ww_models.ModelSingleIndustry(N=N,
                                  theta=theta,
                                  rho=rho,
                                  tau=np.random.rand(N,N,N)/2 + 1,
                                  a = np.random.rand(N,N,N)/2 + 1,
                                  g = np.random.rand(N,N,N)/2 + 1,
                                  L=np.ones(N),
                                  D=np.zeros(N),
                                  )
    truemodel.solve()

    # Generate data
    df_M = pd.DataFrame(data=np.sum(truemodel.X,axis=2),index=set_use,columns=set_use)
    df_T = pd.DataFrame(data=np.sum(truemodel.X,axis=0),index=set_use,columns=set_use)
    df_Zl = pd.DataFrame(data=np.sum(truemodel.Z,axis=(0,2)),index=set_use,columns=["Zl"])

    logger.debug(df_M.head())
    logger.debug(df_T.head())
    logger.debug(df_Zl.head())

    # Save DataFrame to excel sheet
    with pd.ExcelWriter(result_dir+"/TM.xlsx", engine='openpyxl') as writer:    
        df_T.to_excel(writer,sheet_name='T')
        df_M.to_excel(writer,sheet_name='M')
        df_Zl.to_excel(writer,sheet_name='Zl')

    #% Translate data to XML file for NEOS server
    data = open(result_dir + "/data.xml","w") 
    data.write("<data><![CDATA[")
    data.write("set COU := " + " ".join(set_use) + ";")
    data.write("\n")
    data.write( "param mprod : " + " ".join(list(set_use)) + " := \n" 
                              + df_M.to_string(header=False,index_names=False) + ';') 
    data.write("\n")
    data.write( "param trade : " + " ".join(list(set_use)) + " := \n" 
                              + df_T.to_string(header=False,index_names=False) + ';')
    data.write("\n")
    data.write( "param Zl := \n " 
                              + df_Zl.to_string(header=False,index_names=False) + ';')
    data.write("\n")
    data.write("]]></data>")
    data.close() 

def calc_GOGTGM_SpecialCases(result_dir,set_use,theta,rho,logger):
    return

def two_country_example(result_dir,theta,rho,logger):
    return

def calc_GO(result_dir,set_use,theta,rho,logger):
    return

def calc_ExactHatAlgebra_SpecialCases(result_dir,set_use,theta,rho,logger):
    return

def calc_ExactHatAlgebra(result_dir,set_use,theta,rho,logger):
    return

if __name__ == "__main__":
    main_test()