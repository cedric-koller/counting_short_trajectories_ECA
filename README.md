# Counting Short Trajectories in Elementary Cellular Automata using the Transfer Matrix Method
This repositery contains the code to reproduce the results and figure from the work above:
- space-time-diagrams.ipynb is used to generate the space-time diagrams figures.
- generate_compatible_permutations.ipynb is used to create and save arrays containing the allowed indexes. As these indexes are the same for every rule, it should only be done once, and the results are saved in the folder indexes_compatible_permutations.
- simulation.ipynb is used to create table 1 from the paper and the data for the entropy plots. The arrays containing the allowed indexes for the appropriate T=p+c should be created beforehand.
- plots.ipynb is used to create entropy plots and generate the table in latex format.
- The src folder contains the source code for the dynamics transfer matrix for ECAs, the dynamics simulations of the ECA and visualization functions.
- table_entropy.csv contains the resulting entropy for all ECA rules for various p,c and neighborhoods.
- s_given_rho_rule_128.csv and s_given_rho_rule_128_ending_all_0.csv contain the data for figure 4.

