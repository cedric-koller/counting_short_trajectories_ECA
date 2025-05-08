# Counting Short Trajectories in Elementary Cellular Automata using the Transfer Matrix Method
This repositeroy contains the code to reproduce the results and figure from the work above:
- space-time-diagrams.ipynb is used to generate the space-time diagrams figures.
- generate_compatible_permutations.py is used to create and save arrays containing the allowed indexes. As these indexes are the same for every rule, it should only be done once, and the results are saved in the folder indexes_compatible_permutations.
- simulations.ipynb is used to create table_entropy.csv as well as the data for figure 4 from the paper. The numpy arrays containing the allowed indexes for the appropriate T should be created beforehand.
- plots.ipynb is used to create entropy vs p plots and generate the table in latex format from the data of table_entropy.csv
- The src folder contains the source code for the dynamics transfer matrix for ECAs, the dynamics simulations of the ECA and visualization functions.
- the figures folder contains all the figures from the paper
- table_entropy.csv contains the entropy for all the rules and possible combinations of transient length p, cycle length c and neighborhoods.

