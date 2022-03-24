This folder contains file required for simulating the developement of Breast Cancer in PyMT-MMTV mouse models without immune cells influx.

Corresponding paper: "A Bio-Mechanical PDE model of breast tumor progression in MMTV-PyMT mice"

Code Authors: Navid Mohammad Mirzaei (https://github.com/nmirzaei) (https://sites.google.com/view/nmirzaei)

Executable file: Main.py

Requirements:

1- FEniCS version : 2019.2.0.dev0

2- pandas, scipy, numpy

3- Mesh files are needed for this folder to run. Please download them from "A-Bio-Mechanical-PDE-model-of-breast-tumor-progression-in-MMTV-PyMT-mice/Meshes"

4- For remeshing a legacy version of GMSH needs to be installed (i.e. gmsh-3.0.6-Linux64). In addition, it should be callable from within FEniCS. Use the          following lines to install is on Docker:

        /bin/sh -c "sudo apt-get update; sudo apt-get install -y libgl1-mesa-glx libxcursor1 libxft2 libxinerama1 libglu1-mesa"
        wget -nc --quiet gmsh.info/bin/Linux/gmsh-3.0.6-Linux64.tgz
        tar -xf gmsh-3.0.6-Linux64.tgz
        sudo cp -r gmsh-3.0.6-Linux64/share/* /usr/local/share/
        sudo cp gmsh-3.0.6-Linux64/bin/* /usr/local/bin
        
