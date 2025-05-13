""" The main function for creating the bifurcation diagram
    
    Note: run either bifurcationdiagram or cleanedbifurcationdiagram, not both.
    
    Running bifurcationdiagram creates a category_matrix.txt where the 
    categories for each gridpoint in the diagram is saved as a value from
    0 to 5, where 
        0: "Chaos (a)",
        1: "Chaos and Cyclic Points (b)",
        2: "Quasi-Periodic Orbit (c)",
        3: "Quasi-Periodic Orbit and Cyclic Points (d)",
        4: "Multi-Year-Cycles (e)",
        5: "One Year Cycle (f)"
    
    If there is need for manual correction of some misclassified grid values,
    one needs to run cleanedbifurcationdiagram. 
    In order to run this function, one needs to copy category_matrix.txt and 
    save it as category_matrix_cleaned.txt. Then, the misclassified gridcells
    can be corrected by changing the value at the according spot in the matrix.
    
    After everything is classified correctly and saved in 
    category_matrix_cleaned.txt, cleanedbifurcationdiagram can be run.
"""

import os
from bifurcation import bifurcationdiagram, cleanedbifurcationdiagram

base_directory = os.getcwd()

#bifurcationdiagram(base_directory)
cleanedbifurcationdiagram(base_directory)