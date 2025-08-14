""" The main function for creating multiple grid simulations of the clustered poincare maps.
"""

import os
import matplotlib.pyplot as plt
import time

from poincaregrid import newpoincgrid
from clusteringfunctions import clustering
from csv_functions import save_list_to_csv
from seasonal_odes import np_odes, getr

growth_rate = getr()

# Define the range of values for aS and nu
# test case:
aS_values = [2*(0.5+i*0.01) for i in range(0, 6)] # values for aS from 0.5 to 0.55
nu_values = [0.1 + 0.1*i for i in range(0, 5)] # values for nu from 0.1 to 0.5
# full analysis:
#aS_values = [2*(0.5+i*0.01) for i in range(0, 51)] # values for aS from 0.5 to 1.0
#nu_values = [0.1 + 0.1*i for i in range(0, 60)] # values for nu from 0.1 to 6.0

# Define the current directory in order to save the results later on
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# Iterate over the values of aS and nu
for nu in nu_values:
    nu_folder = os.path.join(parent_directory, f"nu_{nu:.2f}")
    os.makedirs(nu_folder, exist_ok=True)
    
    startnewnutime = time.time()
    
    for aS in aS_values:
        aS_folder = os.path.join(nu_folder, f"aS_{aS:.2f}")
        os.makedirs(aS_folder, exist_ok=True)
        
        plt.close('all')
        
        print(aS, nu)
        starttime = time.time()
        
        # Run the clustering function (assuming newpoincgrid and clustering functions are defined)
        sol = newpoincgrid(np_odes, 5000, 0.01, aS, nu, 0.01, 1.0, 0.01, 0.2, 10, 10)
        intermediatetime = time.time()
        elapsedtime = intermediatetime - starttime
        timemins = int(elapsedtime // 60)
        timesecs = elapsedtime % 60
        print(f"Simulating the grid took {timemins} minutes and {timesecs} seconds")
        
        output = clustering(np_odes, sol, aS, nu, 1, True, aS_folder)
        
        # Save the lists to CSV files
        save_list_to_csv(aS_folder, "clusterlist", output[2])
        save_list_to_csv(aS_folder, "chaoslist", output[3])
        save_list_to_csv(aS_folder, "cyclelist", output[4])
        save_list_to_csv(aS_folder, "notsurelist", output[5])
        
        endtime = time.time()
        elapsedtime = endtime - starttime
        timemins = int(elapsedtime // 60)
        timesecs = elapsedtime % 60
        print(f"The simulation for aS={aS} and nu={nu} took {timemins} minutes and {timesecs} seconds.")

    endnutime = time.time()
    elapsednutime = endnutime - startnewnutime
    timemins_nu = int(elapsednutime // 60)
    timesecs_nu = elapsednutime % 60
    print(f"The simulation for all aS for nu={nu} took {timemins_nu} minutes and {timesecs_nu} seconds.")

print("The outputs have been saved to respective folders.")