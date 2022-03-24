'''
***Converts GMSH files into .xdmf***

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
###############################################################

###############################################################
#Read .xml mesh files
###############################################################
mesh = Mesh("Mesh.xml")
Volume = MeshFunction("size_t", mesh, "Mesh_physical_region.xml")
bnd_mesh = MeshFunction("size_t", mesh, "Mesh_facet_region.xml")
###############################################################

###############################################################
#Convert them to .xdmf files
###############################################################
xdmf = XDMFFile(mesh.mpi_comm(),"Mesh1.xdmf")
xdmf.write(mesh)
xdmf.write(Volume)
xdmf = XDMFFile(mesh.mpi_comm(),"boundaries1.xdmf")
xdmf.write(bnd_mesh)
xdmf.close()
###############################################################
