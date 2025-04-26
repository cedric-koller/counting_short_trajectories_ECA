# Counting Short Trajectories in Elementary Cellular Automata using the Transfer Matrix Method
This repositeroy contains the code to reproduce the results and figure from the work above:
- space-time-diagrams.ipynb is used to generate the space-time diagrams figures.
- generate_compatible_permutations.py is used to create and save arrays containing the allowed indexes. As these indexes are the same for every rule, it is done only once, and the results are saved in the folder indexes_compatible_permutations up to T=10.
- create_table.py is used to create table 1 from the paper.
- plots.ipynb is used to create entropy vs p plots.
- entropy_vs_density.ipynb is used to compute the data and generate figure TODO
- The src folder contains the source code for the dynamics transfer matrix for ECAs, the dynamics simulations of the ECA and visualization functions.

