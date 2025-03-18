#Lod_Code_nonmonotone_nonlinearPDE
This repository contains the LOD code for the experiments in the paper "Linearized Localized Orthogonal Decomposition for Quasilinear Nonmonotonic Elliptic PDE." The implementation is divided
into two parts. One part performs the experiments for the Kacanov-type linearization, the other for the linearization of the Fr'echet derivative. The correction calculations were implemented using 
parallelization code.
All numerical results of the paper can be reproduced by running MainCodeCacanove or MainCodeFrechet files:
The fine mesh size Nh can be adjusted. The "p" in the for loop of the conver_history function represents different sizes of the coarse mesh size NH.
To test different sizes of localization parameters, the "i" in global_exper(i) can be adjusted.
