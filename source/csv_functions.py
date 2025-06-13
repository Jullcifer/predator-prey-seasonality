""" Here the csv functions for saving and reading are specified.
"""

import csv
import os

def save_list_to_csv(folder_path, list_name, data_list):
    """ Saving the output of the clustering simulation in the respective lists in a nice format
    
    Args:
        folder_path: the path where our folder is located, where the file should be saved
        list_name: the name of our file - i.e. either 'clusterlist' or any of the other categories.
        data_list: the output of the clustering simulation. Is of a different form, depending on list_name:
        
            Note: the output is of a different form if we have clusterlist than in all the other categories.
            If we have clusterlist, we have entries of the form [listindex, vector of (n,p) coordinates, length of the cyclic point]
            Otherwise, we just have entries with the [n,p] coordinates (as we either have a cycle or chaos)
    """
    
    if data_list:
        file_path = os.path.join(folder_path, f"{list_name}.csv")
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if list_name == "clusterlist":
                writer.writerow(["listindex", "n", "p", "cyclelength"])
                for item in data_list:
                    writer.writerow([item[0], item[1][0], item[1][1], item[2]])
            else:
                writer.writerow(["n", "p"])
                for item in data_list:
                    writer.writerow([item[0], item[1]])



def read_csv(file_path):
    """Reading in data from csv files
    
    Args:
        file_path: the path where our file is located
    
    Returns:
        data: array with all the rows in the csv file
    """
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        data = [row for row in reader]
    return data