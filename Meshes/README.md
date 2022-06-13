Corresponding paper: "A PDE Model of Breast Tumor Progression in MMTV-PyMT Mice"

Code Authors: Navid Mohammad Mirzaei (https://github.com/nmirzaei) (https://sites.google.com/view/nmirzaei)

Executable file: HDF5Convert.py

Requirements:

1- FEniCS version : 2019.2.0.dev0

3- The .xml files are needed for this folder to run. To get them you should have a legacy version of GMSH which is compatible with Fenics (i.e. gmsh-3.0.6-Linux64). After creating .geo file and saving it through GMSH you will have a .msh file. Then on a terminal type

        dolfin-convert Mesh.msh Mesh.xml
        
This will create the .xml file which will then be used by "HDF5Convert.py" to create the .xdmf files
