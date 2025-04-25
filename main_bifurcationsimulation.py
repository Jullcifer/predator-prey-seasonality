""" The main function for creating the bifurcation diagram
"""

import os
from bifurcation import bifurcationdiagram, cleanedbifurcationdiagram

base_directory = os.getcwd()

# do either bifurcationdiagram or cleanedbifurcationdiagram.
# cleanedbifurcationdiagram can only be run after we already have run
# bifurcationdiagram once and cleaned the results manually.

#bifurcationdiagram(base_directory)
cleanedbifurcationdiagram(base_directory)