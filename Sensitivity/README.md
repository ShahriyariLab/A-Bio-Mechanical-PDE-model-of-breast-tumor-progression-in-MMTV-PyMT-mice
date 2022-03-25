This folder contains file required for sensitivity analysis

Corresponding paper: "A Bio-Mechanical PDE model of breast tumor progression in MMTV-PyMT mice"

Code Authors: Navid Mohammad Mirzaei (https://github.com/nmirzaei) (https://sites.google.com/view/nmirzaei)


Requirements:

1- FEniCS version : 2019.2.0.dev0

2- For these folder we need to pyadjoint as well. Since all the sensitivity analyses uses features from this package: (https://www.dolfin-adjoint.org/en/latest/download/index.html)

2- pandas, scipy, numpy

3- Mesh files are needed for this folder to run. Please download them from "A-Bio-Mechanical-PDE-model-of-breast-tumor-progression-in-MMTV-PyMT-mice/Meshes"

Run order:

Since pyadjoint occupies a lot of memory it is recommended you find the displacements separately. Use the following order:

1- Run Sensitivity/Displacement/Main.py to get the displacement fields. (Notice that this file does not call pyadjoint and it can be done via fenics alone)

2- Now you should have a displacement folder with all the displacement fields created from part 1. Copy it in the ODE_parameters and PDE_parameters folders.

3- Proceed with running the codes in ODE_parameters and PDE_parameters folder using a pyadjoint inegrated FEniCS. This step can take a while. 
