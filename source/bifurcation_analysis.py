import numpy as np
from data_processing import classify_and_read_data
from bifurcation_plot import plot_bifurcation_diagram

def bifurcationdiagram(base_directory, saveplot=True):
    """
    This function uses the results from the classification to create the 
    bifurcation diagram. It uses the uncleaned results, i.e. with the 
    notsure component and potential misclassifications.
    Along the path, a category_matrix is being saved, allowing us to
    easily clean the data for the cleaned bifurcation diagram.

    Args:
        base_directory (str): Directory where the data is stored.
        saveplot (bool): Whether to save the plot. Defaults to True.
    """
    simulation_data = classify_and_read_data(base_directory)
    a_s_values, nuvalues, categoryvalues = [], [], []
    for nu, a_s_data in simulation_data.items():
        for a_s, details in a_s_data.items():
            a_s_val = float(a_s.split('_')[1])
            nu_val = float(nu.split('_')[1])
            a_s_values.append(np.round(a_s_val, 2))
            nuvalues.append(np.round(nu_val, 2))
            categoryvalues.append(details['categoryvalues'])
    a_s_unique = np.unique(a_s_values)
    nu_unique = np.unique(nuvalues)
    category_matrix = np.full((len(nu_unique), len(a_s_unique)), np.nan)
    for a_s, nu, category in zip(a_s_values, nuvalues, categoryvalues):
        i = np.where(nu_unique == nu)[0][0]
        j = np.where(a_s_unique == a_s)[0][0]
        category_matrix[i, j] = category
    np.savetxt("category_matrix.txt", category_matrix, fmt='%.0f', delimiter=', ')
    if saveplot:
        plot_bifurcation_diagram(a_s_unique, nu_unique, category_matrix, f"{base_directory}/BifurcationGridPlot.png")