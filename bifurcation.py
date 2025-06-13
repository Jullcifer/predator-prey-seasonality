""" This file contains 3 functions which are used to create the bifurcation
    diagram from our simulation results.
    
    classify_and_read_data and bifurcationdiagram use the simulation results 
    directly, whereas bifurcationdiagram_cleaned requires a 
    manually cleaned data set, saved as category_matrix_cleaned.txt which needs 
    to be modified as a copy of category_matrix.txt which originates from the 
    bifurcationdiagram function.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from csv_functions import read_csv


def classify_and_read_data(base_dir):
    """ This function maps the results from the long-time simulations for all the 
        combinations of aS and nu to the categories resulting from that. 
        The categories are classified by numbers 0 to 9, where 
                0: "Only chaos",
                1: "Chaos and notsure",
                2: "Chaos, notsure and cyclic points",
                3: "Chaos and cyclic points",
                4: "Only cycle",
                5: "Cycle and notsure",
                6: "Cycle and cyclic points",
                7: "Cyclic points and notsure",
                8: "Only one (yearly) cyclic point",
                9: "Only cyclic points"

    Args: 
        base_dir: our current working directory, i.e. where the data is stored
        
    Returns:
        data: a dictionary mapping all the parameter combinations (aS, nu) to the categories resulting from the simulations for the corresponding aS and nu

    """
    data = {}
    
    for nu_dir in os.listdir(base_dir):
        # skipping the folder if it is not a nu-folder
        if not nu_dir.startswith("nu"):
            continue
        
        nu_path = os.path.join(base_dir, nu_dir)
        if os.path.isdir(nu_path):
            data[nu_dir] = {}
            
            for aS_dir in os.listdir(nu_path):
                aS_path = os.path.join(nu_path, aS_dir)
                if os.path.isdir(aS_path):
                    data[nu_dir][aS_dir] = {}
                    
                    lists_data = {}
                    for list_name in ["clusterlist", "chaoslist", "cyclelist", "notsurelist"]:
                        file_path = os.path.join(aS_path, f"{list_name}.csv")
                        if os.path.exists(file_path):
                            lists_data[list_name] = read_csv(file_path)
                        else:
                            lists_data[list_name] = []

                    # Determine categoryvalues based on the given conditions
                    '''
                    categoryvalues = None
                    if lists_data["chaoslist"] and not lists_data["clusterlist"] and not lists_data["cyclelist"] and not lists_data["notsurelist"]:
                        categoryvalues = 0
                    elif lists_data["chaoslist"] and not lists_data["clusterlist"] and not lists_data["cyclelist"] and lists_data["notsurelist"]:
                        categoryvalues = 1
                    elif lists_data["chaoslist"] and lists_data["clusterlist"] and not lists_data["cyclelist"] and lists_data["notsurelist"]:
                        #categoryvalues = 2
                        categoryvalues = 3
                    elif lists_data["chaoslist"] and lists_data["clusterlist"] and not lists_data["cyclelist"] and not lists_data["notsurelist"]:
                        categoryvalues = 3
                    elif not lists_data["chaoslist"] and lists_data["clusterlist"] and not lists_data["cyclelist"] and lists_data["notsurelist"]:
                        #categoryvalues = 7
                        categoryvalues = 6
                    elif not lists_data["chaoslist"] and not lists_data["clusterlist"] and lists_data["cyclelist"] and not lists_data["notsurelist"]:
                        categoryvalues = 4
                    elif not lists_data["chaoslist"] and not lists_data["clusterlist"] and lists_data["cyclelist"] and lists_data["notsurelist"]:
                        #categoryvalues = 5
                        categoryvalues = 4
                    elif not lists_data["chaoslist"] and lists_data["clusterlist"] and lists_data["cyclelist"] and not lists_data["notsurelist"]:
                        categoryvalues = 6
                    elif not lists_data["chaoslist"] and len(lists_data["clusterlist"]) == 1:
                        categoryvalues = 8
                    elif not lists_data["chaoslist"] and len(lists_data["clusterlist"]) > 1:
                        categoryvalues = 9
                    '''
                    categoryvalues = None
                    if lists_data["chaoslist"] and not lists_data["clusterlist"] and not lists_data["cyclelist"]:
                        # Chaos (a)
                        categoryvalues = 0
                    elif lists_data["chaoslist"] and lists_data["clusterlist"] and not lists_data["cyclelist"]:
                        # Chaos and Cyclic Points (b)
                        categoryvalues = 1
                    elif not lists_data["chaoslist"] and not lists_data["clusterlist"] and lists_data["cyclelist"]:
                        # Quasi-Periodic Orbit (c)
                        categoryvalues = 2
                    elif not lists_data["chaoslist"] and lists_data["clusterlist"] and lists_data["cyclelist"]:
                        # Quasi-Periodic Orbit and Cyclic Points (d)
                        categoryvalues = 3
                    elif not lists_data["chaoslist"] and not lists_data["cyclelist"] and len(lists_data["clusterlist"]) == 1:
                        # One Year Cycle (f)
                        categoryvalues = 5
                    elif not lists_data["chaoslist"] and not lists_data["cyclelist"] and len(lists_data["clusterlist"]) > 1:
                        # Multi-Year Cycles (e)
                        categoryvalues = 4

                    data[nu_dir][aS_dir] = {
                        "lists": lists_data,
                        "categoryvalues": categoryvalues
                    }
    
    return data


def bifurcationdiagram(base_directory, saveplot=True):
    """ This function uses the results from the classification to create the 
        bifurcation diagram. It uses the uncleaned results, i.e. with the 
        notsure component and potential misclassifications.
        Along the path, a category_matrix is being saved, allowing us to
        easily clean the data for the cleaned bifurcation diagram.

    Args: 
        base_dir: our current working directory, i.e. where the data is stored
        saveplot: boolean to determine whether we want to save the diagram

    """
    simulation_data = classify_and_read_data(base_directory)
    
    aSvalues = []
    nuvalues = []
    categoryvalues = []
    
    # Print the data to verify
    for nu, aS_data in simulation_data.items():
        for aS, details in aS_data.items():
            print(aS.split('_'))
            aS_val = aS.split('_')[1]
            print(nu.split('_'))
            nu_val = nu.split('_')[1]
            print(f"nu: {nu_val}, aS: {aS_val}")
            aSvalues.append(np.round(float(aS_val), 2))
            nuvalues.append(np.round(float(nu_val), 2))
            print(f"  categoryvalues: {details['categoryvalues']}")
            categoryvalues.append(details['categoryvalues'])
            for list_name, data in details['lists'].items():
                if data:
                    print(f"  {list_name}: {len(data)} entries")
                else:
                    print(f"  {list_name}: No data")
    
    cmap_categories = plt.get_cmap('rainbow', 6)
    
    
    # Create a grid of aS and nu values
    aS_unique = np.unique(aSvalues)
    nu_unique = np.unique(nuvalues)
    
    # Create a matrix to hold the category values
    category_matrix = np.full((len(nu_unique), len(aS_unique)), np.nan)
    
    # Fill the matrix with category values
    for aS, nu, category in zip(aSvalues, nuvalues, categoryvalues):
        i = np.where(nu_unique == nu)[0][0]
        j = np.where(aS_unique == aS)[0][0]
        category_matrix[i, j] = category
        
    # Save the category_matrix to a text file
    np.savetxt("category_matrix.txt", category_matrix, fmt='%.0f', delimiter=', ')
        
    
    category_labels = {
        0: "Chaos (a)",
        1: "Chaos and Cyclic Points (b)",
        2: "Quasi-Periodic Orbit (c)",
        3: "Quasi-Periodic Orbit and Cyclic Points (d)",
        4: "Multi-Year-Cycles (e)",
        5: "One Year Cycle (f)"
    }
    
    # Now the plotting
    # Set font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # Create the plot with squares/rectangles filling out the whole space
    fig, ax = plt.subplots(figsize=(10, 6), dpi = 150)
    plt.pcolormesh(aS_unique, nu_unique, category_matrix, cmap=cmap_categories, shading='auto', vmin=0, vmax=5)
   
    # Add a colorbar with text labels
    cbar = plt.colorbar()
    cbar.set_label("Category Descriptions", fontsize = 8, labelpad = 5)
    
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels([category_labels[i] for i in [0, 1, 2, 3, 4, 5]])
   
    # Add labels and title
    plt.xlabel(r'$a_S$', fontsize = 12)
    plt.ylabel(r'$\nu$', fontsize = 12)
    plt.title("Numerical Bifurcation Diagram", fontsize = 10, pad = 5)
    
    # To avoid cutting off the legend
    plt.tight_layout()
    
    # Make additional adjustments to get some space on the boundaries
    # plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
   
    # Show the plot
    plt.show()
    
    # If we want to save the plot
    if saveplot == True:
        fig.savefig(f"{base_directory}/BifurcationGridPlot.png", bbox_inches='tight')
    
    #print(category_matrix)
    
    return()
    

def cleanedbifurcationdiagram(base_directory, saveplot=True):
    """ This function uses the results from the classification to create the 
        bifurcation diagram. It uses the manually cleaned results, i.e. gives 
        us the final, cleaned bifurcation diagram.

    Args: 
        base_dir: our current working directory, i.e. where the data is stored
        saveplot: boolean to determine whether we want to save the diagram

    """
    simulation_data = classify_and_read_data(base_directory)

    aSvalues = []
    nuvalues = []
    categoryvalues = []

    # Print the data to verify
    for nu, aS_data in simulation_data.items():
        for aS, details in aS_data.items():
            aS_val = aS.split('_')[1]
            nu_val = nu.split('_')[1]
            print(f"nu: {nu_val}, aS: {aS_val}")
            aSvalues.append(np.round(float(aS_val), 2))
            nuvalues.append(np.round(float(nu_val), 2))
            print(f"  categoryvalues: {details['categoryvalues']}")
            categoryvalues.append(details['categoryvalues'])
            for list_name, data in details['lists'].items():
                if data:
                    print(f"  {list_name}: {len(data)} entries")
                else:
                    print(f"  {list_name}: No data")

    cmap_categories = plt.get_cmap('rainbow', 6)


    # Create a grid of aS and nu values
    aS_unique = np.unique(aSvalues)
    nu_unique = np.unique(nuvalues)
    
    # Define the text labels for each category
    category_labels_cleaned = {
        0: "Chaos (a)",
        1: "Chaos and Cyclic Points (b)",
        2: "Quasi-Periodic Orbit (c)",
        3: "Quasi-Periodic Orbit and Cyclic Points (d)",
        4: "Multi-Year-Cycles (e)",
        5: "One Year Cycle (f)"
    }


    # Read the content of the file
    with open('category_matrix_cleaned.txt', 'r') as file:
        input_str = file.read()

    # Split the string into rows based on the newline character
    rows = input_str.strip().split('\n')

    # Initialize an empty list to store the converted rows
    category_matrix = []

    # Iterate over each row
    for row in rows:
        # Split the row into individual elements based on commas
        elements = row.split(',')
        
        # Convert each element to an integer and add it to the array_2d list
        converted_row = [int(float(element)) for element in elements]
        category_matrix.append(converted_row)
        
    # Now the plotting
    # Set font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # Create the plot with squares/rectangles filling out the whole space
    fig, ax = plt.subplots(figsize=(14, 9), dpi = 600)
    plt.pcolormesh(aS_unique, nu_unique, category_matrix, cmap=cmap_categories, shading='auto', vmin=0, vmax=5)
   
    # Add a colorbar with text labels
    cbar = plt.colorbar()
    cbar.set_label('Category Descriptions', fontsize = 14, labelpad = 10)
    
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels([category_labels_cleaned[i] for i in [0, 1, 2, 3, 4, 5]])
   
    # Add labels and title
    plt.xlabel(r'$a_S$', fontsize = 14)
    plt.ylabel(r'$\nu$', fontsize = 14)
    plt.title('Numerical Bifurcation Diagram', fontsize = 18, pad = 20)
    
    # To avoid cutting off the legend
    plt.tight_layout()
    
    # Make additional adjustments to get some space on the boundaries
    plt.subplots_adjust(left=0.05, right=0.8, top=0.9, bottom=0.1)
   
    # Show the plot
    plt.show()
    
    # Saving the plot if desired
    if saveplot == True:
        fig.savefig(f"{base_directory}/BifurcationGridPlot_Cleaned.png")
    
    return()