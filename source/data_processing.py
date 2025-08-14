import os
from csv_functions import read_csv

def classify_and_read_data(base_dir):
    """
    This function maps the results from the long-time simulations for all the 
    combinations of aS (summer length) and nu (density-dependence) to 
    the categories resulting from that. 
    The categories are classified by numbers from 0 to 5, where: 
        0: "Chaos (a)",
        1: "Chaos and Cyclic Points (b)",
        2: "Quasi-Periodic Orbit (c)",
        3: "Quasi-Periodic Orbit and Cyclic Points (d)",
        4: "Multi-Year-Cycles (e)",
        5: "One Year Cycle (f)"

    Args:
        base_dir (str): Directory where the simulation data is stored.

    Returns:
        data: A dictionary mapping all the parameter combinations (aS, nu) 
        to the categories resulting from the simulations for the corresponding 
        aS and nu

    """
    data = {}
    for nu_dir in os.listdir(base_dir):
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
                    for list_name in ["clusterlist", "chaoslist", "cyclelist"]:
                        file_path = os.path.join(aS_path, f"{list_name}.csv")
                        lists_data[list_name] = read_csv(file_path) if os.path.exists(file_path) else []
                    # Determine category values based on conditions
                    categoryvalues = None
                    if lists_data["chaoslist"] and not lists_data["clusterlist"] and not lists_data["cyclelist"]:
                        categoryvalues = 0  # Chaos
                    elif lists_data["chaoslist"] and lists_data["clusterlist"] and not lists_data["cyclelist"]:
                        categoryvalues = 1  # Chaos and Cyclic Points
                    elif not lists_data["chaoslist"] and not lists_data["clusterlist"] and lists_data["cyclelist"]:
                        categoryvalues = 2  # Quasi-Periodic Orbit
                    elif not lists_data["chaoslist"] and lists_data["clusterlist"] and lists_data["cyclelist"]:
                        categoryvalues = 3  # Quasi-Periodic Orbit and Cyclic Points
                    elif not lists_data["chaoslist"] and not lists_data["cyclelist"] and len(lists_data["clusterlist"]) == 1:
                        categoryvalues = 5  # One Year Cycle
                    elif not lists_data["chaoslist"] and not lists_data["cyclelist"] and len(lists_data["clusterlist"]) > 1:
                        categoryvalues = 4  # Multi-Year Cycles
                    data[nu_dir][aS_dir] = {
                        "lists": lists_data,
                        "categoryvalues": categoryvalues
                    }
    return data


