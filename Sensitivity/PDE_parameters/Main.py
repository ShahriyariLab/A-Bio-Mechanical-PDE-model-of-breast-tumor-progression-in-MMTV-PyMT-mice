'''
A Bio-Mechanical PDE model of breast tumor progression in MMTV-PyMT mice

**Calculates sensitivities of total cancer cell to PDE Parameters**

IC_Loc_DG: Creating spatially distribute initial conditions based on biological
            data

Smoothen: Smoothens discountinous data that are projected onto a continuous space

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
#This has to be installed separately from Fenics
from dolfin_adjoint import *
###############################################################
import logging
import scipy.optimize as op
import pandas as pd
from subprocess import call
from IC_Loc_DG import *
from ufl import Min
from smoothen import *
####################################

###############################################################
#Solver parameters
###############################################################
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
###############################################################

###############################################################
#Remeshing info
###############################################################
#For no refinement set refine=1 for refinement set the intensity >1
#Org_size is the original element size of your mesh. Can be extracted from the .geo mesh files
#Max_cellnum and Min_cellnum will assure that after remeshing the number of new cells are in [Min_cellnum,Max_cellnum]
#remesh_step is the step when remeshing is initiated. Set it equal to negative or very large number for no remeshing
refine = 1
Refine = str(refine)
Org_size = 0.0026
Max_cellnum = 3400
Min_cellnum = 3300
remesh_step = 3500
###############################################################

###############################################################
#Reporting options
###############################################################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
###############################################################


###############################################################
# Nullspace of rigid motions
###############################################################
#Translation in 3D. Comment out if the problem is 2D
#Z_transl = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]

#Rotations 3D. Comment out if the problem is 2D
#Z_rot = [Expression(('0', 'x[2]', '-x[1]')),
#         Expression(('-x[2]', '0', 'x[0]')),
#         Expression(('x[1]', '-x[0]', '0'))]

##Translation 2D. Comment out if the problem is 3D
Z_transl = [Constant((1, 0)), Constant((0, 1))]

# Rotations 2D. Comment out if the problem is 3D
Z_rot = [Expression(('-x[1]', 'x[0]'),degree=0)]
# All
Z = Z_transl + Z_rot
###############################################################


###############################################################
# Load mesh
###############################################################
#Parallel compatible Mesh readings
mesh= Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), "Mesh.xdmf")
xdmf.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("Mesh.xdmf") as infile:
    infile.read(mvc, "f")
Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
mvc2 = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("boundaries.xdmf") as infile:
    infile.read(mvc2, "f")
bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
###############################################################

###############################################################
#Saving the mesh files
###############################################################
File('Results/mesh.pvd')<<mesh
File('Results/Volume.pvd')<<Volume
File('Results/boundary.pvd')<<bnd_mesh
###############################################################

###############################################################
# Build function spaces
###############################################################
#Mechanical problem
P22 = VectorElement("P", mesh.ufl_cell(), 4)
P11 = FiniteElement("P", mesh.ufl_cell(), 1)
P00 = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
TH = MixedElement([P22,P11,P00])
W = FunctionSpace(mesh, TH)

#Biological problem
#Nodal Enrichment is done for more stability
P1 = FiniteElement('P', triangle,1)
PB = FiniteElement('B', triangle,3)
NEE = NodalEnrichedElement(P1, PB)
element = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
Mixed_Space = FunctionSpace(mesh, element)

#Auxillary spaces for projection and plotting
SDG = FunctionSpace(mesh,'DG',0)
S1 = FunctionSpace(mesh,'P',1)
VV = VectorFunctionSpace(mesh,'Lagrange',4)
VV1 = VectorFunctionSpace(mesh,'Lagrange',1)
R = FunctionSpace(mesh,'R',0)
###############################################################

###############################################################
#Defining functions and test functions
###############################################################
U = Function(Mixed_Space)
U_n = Function(Mixed_Space)
Tn,Th,Tc,Tr,Dn,D,Mn,M,C,N,A,H,IL12,IL10,IL6 = split(U)
Tn_n,Th_n,Tc_n,Tr_n,Dn_n,D_n,Mn_n,M_n,C_n,N_n,A_n,H_n,IL12_n,IL10_n,IL6_n= split(U_n)
v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15= TestFunctions(Mixed_Space)
u_n = Function(VV)
UU = Function(W)
u_, p_, lambda_ =split(UU)
u__ = Function(VV1)
displ = Function(VV)
#######################################################################

# Construct integration measure using these markers
ds = Measure('ds', subdomain_data=bnd_mesh)
dx = Measure('dx', subdomain_data=Volume)
###############################################################


###############################################################
#time step variables
###############################################################
T = 3501            # final
num_steps= 3501     # number of time steps
dt = T / num_steps  # time step
eps = 1             # diffusion coefficient
t=0                 # initial time
k = Constant(dt)    # Constant tip step object for weak formulation
###############################################################

###############################################################
# reading parameter values from file
###############################################################
parameters = pd.read_csv('input/input/parameters.csv').to_numpy()
params = parameters[0]
###############################################################

###############################################################
#ODE Parameters non-dimensional
###############################################################
Pars =['\lambda_{T_hH}','\lambda_{T_hD}','\lambda_{T_hIL_{12}}',
                        '\lambda_{T_cD}','\lambda_{T_cIL_{12}}',
                        '\lambda_{T_rD}',
                        '\lambda_{DC}','\lambda_{DH}',
                        '\lambda_{MIL_{10}}','\lambda_{MIL_{12}}','\lambda_{MT_h}',
                        '\lambda_{C}','\lambda_{CIL_6}','\lambda_{CA}',
                        '\lambda_{A}',
                        '\lambda_{HD}','\lambda_{HN}','\lambda_{HM}','\lambda_{HT_c}','\lambda_{HC}',
                        '\lambda_{IL_{12}M}','\lambda_{IL_{12}D}','\lambda_{IL_{12}T_h}','\lambda_{IL_{12}T_c}',
                        '\lambda_{IL_{10}M}','\lambda_{IL_{10}D}','\lambda_{IL_{10}T_r}','\lambda_{IL_{10}T_h}','\lambda_{IL_{10}T_c}','\lambda_{IL_{10}C}',
                        '\lambda_{IL_6A}','\lambda_{IL_6M}','\lambda_{IL_6D}',
                        '\delta_{T_hT_r}','\delta_{T_hIL_{10}}','\delta_{T_h}',
                        '\delta_{T_cIL_{10}}','\delta_{T_CT_r}','\delta_{T_c}',
                        '\delta_{T_r}',
                        '\delta_{T_N}',
                        '\delta_{DC}','\delta_{D}',
                        '\delta_{D_N}',
                        '\delta_{M}',
                        '\delta_{M_N}',
                        '\delta_{CT_c}','\delta_{C}',
                        '\delta_{A}',
                        '\delta_{N}',
                        '\delta_{H}',
                        '\delta_{IL_{12}}',
                        '\delta_{IL_{10}}',
                        '\delta_{IL_6}',
                        'A_{T_N}','A_{D_N}','A_{M}',
                        '\\alpha_{NC}','C_0','A_0']
###############################################################


###############################################################
#Initial conditions
###############################################################
#Tn and Mn are outside of the environment and are modeled as ODEs so uniform IC
#Cytokines are taken to start with a uniform IC. The values extracted from the ODE model
#The rest have been extracted from Biological data through IC_Loc_DG function
Tn_0 = project(Constant(0.1),S1)
Mn_0 = project(Constant(0.1),S1)
H_0 = project(Constant(0.0645577792),S1)
IL12_0 = project(Constant(0.1),S1)
IL10_0 = project(Constant(0.0576763485),S1)
IL6_0 = project(Constant(0.0715311692),S1)
Th_0,Tc_0,Tr_0,Dn_0,D_0,M_0,C_0,N_0,A_0,maxTh,maxTc,maxTr,maxDn,maxD,maxM,maxC,maxN,maxA = IC_Loc_DG(1,mesh,SDG)   #no ICs for molecules
###############################################################

###############################################################
#PDE Parameters dimensional
###############################################################
D_Th, D_Tc, D_Tr, D_Dn, D_D, D_M, D_C, D_N, D_A, D_H, D_IL12, D_IL10, D_IL6 =  Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8), Constant(3.6e-8),Constant(3.3e-3), Constant(5.2e-5) , Constant(5.2e-5) , Constant(5.2e-5)
kappaTh, kappaTc, kappaTr, kappaDn, kappaD, kappaM = Constant(1),Constant(1),Constant(1),Constant(1),Constant(1),Constant(1)
coeff = Constant(1)    #advection constant
Th_R, Tc_R, Tr_R, Dn_R, D_R, M_R= Constant(1/10),Constant(1/10),Constant(1/10),Constant(1/10),Constant(1/10),Constant(1/10)
###############################################################


##############################################################
#Interpolating ICs using a continous Galerkin space
##############################################################
Tn0 = project(Tn_0,S1)
Th0 = project(Th_0,S1)
Tc0 =project(Tc_0,S1)
Tr0 = project(Tr_0,S1)
Dn0 = project(Dn_0,S1)
D0 = project(D_0,S1)
Mn0 = project(Mn_0,S1)
M0 = project(M_0,S1)
C0 = project(C_0,S1)
N0 = project(N_0,S1)
A0 = project(A_0,S1)
H0 = project(H_0,S1)
IL120 = project(IL12_0,S1)
IL100 = project(IL10_0,S1)
IL60 = project(IL6_0,S1)
###############################################################


##############################################################
#Making the negative values caused by coarse projection zero
##############################################################
i_Th=np.argwhere(Th0.vector().get_local()[:]<=0)  #making negatives zero
Th0.vector()[i_Th[:,0]] = 0

i_Tc=np.argwhere(Tc0.vector().get_local()[:]<=0)  #making negatives zero
Tc0.vector()[i_Tc[:,0]] = 0

i_Tr=np.argwhere(Tr0.vector().get_local()[:]<=0)  #making negatives zero
Tr0.vector()[i_Tr[:,0]] = 0

i_Dn=np.argwhere(Dn0.vector().get_local()[:]<=0)  #making negatives zero
Dn0.vector()[i_Dn[:,0]] = 0

i_D=np.argwhere(D0.vector().get_local()[:]<=0)  #making negatives zero
D0.vector()[i_D[:,0]] = 0

i_M=np.argwhere(M0.vector().get_local()[:]<=0)  #making negatives zero
M0.vector()[i_M[:,0]] = 0

i_C=np.argwhere(C0.vector().get_local()[:]<=0)  #making negatives zero
C0.vector()[i_C[:,0]] = 0

i_N=np.argwhere(N0.vector().get_local()[:]<=0)  #making negatives zero
N0.vector()[i_N[:,0]] = 0

i_A=np.argwhere(A0.vector().get_local()[:]<=0)  #making negatives zero
A0.vector()[i_A[:,0]] = 0
##############################################################

##############################################################
#Smoothening the nonzero initial conditions using an initial diffusion
##############################################################
Th0,Tc0,Tr0,Dn0,D0,M0,C0,N0,A0 = smoothen(Th0,Tc0,Tr0,Dn0,D0,M0,C0,N0,A0,S1,mesh)
##############################################################

##############################################################
#Max values of the continuous initial conditions
##############################################################
maxTh0 = max(abs(Th0.vector()[:]))
maxTc0 = max(abs(Tc0.vector()[:]))
maxTr0 = max(abs(Tr0.vector()[:]))
maxDn0 = max(abs(Dn0.vector()[:]))
maxD0 = max(abs(D0.vector()[:]))
maxM0 = max(abs(M0.vector()[:]))
maxC0 = max(abs(C0.vector()[:]))
maxN0 = max(abs(N0.vector()[:]))
maxA0 = max(abs(A0.vector()[:]))
###############################################################

##############################################################
#Using the max values for Non-dimensionalizing. The coefficient 10 is for initial growth calibration
##############################################################
if maxTh0>0:
    Th0.vector()[:]/=10*maxTh0
if maxTc0>0:
    Tc0.vector()[:]/=10*maxTc0
if maxTr0>0:
    Tr0.vector()[:]/=10*maxTr0
if maxDn0>0:
    Dn0.vector()[:]/=10*maxDn0
if maxD0>0:
    D0.vector()[:]/=10*maxD0
if maxM0>0:
    M0.vector()[:]/=10*maxM0
if maxC0>0:
    C0.vector()[:]/=10*maxC0
if maxN0>0:
    N0.vector()[:]/=10*maxN0
if maxA0>0:
    A0.vector()[:]/=10*maxA0
###############################################################


###############################################################
# Projecting the curated ICs onto mixed function space subspaces.
###############################################################
Tn0 = interpolate(interpolate(Tn0,S1), Mixed_Space.sub(0).collapse())
Th0 = interpolate(interpolate(Th0,S1), Mixed_Space.sub(1).collapse())
Tc0 = interpolate(interpolate(Tc0,S1), Mixed_Space.sub(2).collapse())
Tr0 = interpolate(interpolate(Tr0,S1), Mixed_Space.sub(3).collapse())
Dn0 = interpolate(interpolate(Dn0,S1), Mixed_Space.sub(4).collapse())
D0 = interpolate(interpolate(D0,S1), Mixed_Space.sub(5).collapse())
Mn0 = interpolate(interpolate(Mn0,S1), Mixed_Space.sub(6).collapse())
M0 = interpolate(interpolate(M0,S1), Mixed_Space.sub(7).collapse())
C0 = interpolate(interpolate(C0,S1), Mixed_Space.sub(8).collapse())
N0 = interpolate(interpolate(N0,S1), Mixed_Space.sub(9).collapse())
A0 = interpolate(interpolate(A0,S1), Mixed_Space.sub(10).collapse())
H0 = interpolate(interpolate(H0,S1), Mixed_Space.sub(11).collapse())
IL120 = interpolate(interpolate(IL120,S1), Mixed_Space.sub(12).collapse())
IL100 = interpolate(interpolate(IL100,S1), Mixed_Space.sub(13).collapse())
IL60 = interpolate(interpolate(IL60,S1), Mixed_Space.sub(14).collapse())
###############################################################

###############################################################
#Assigning subspace projected ICs as the initial step of iteration
###############################################################
assign(U_n.sub(0),Tn0)
assign(U_n.sub(1),Th0)
assign(U_n.sub(2),Tc0)
assign(U_n.sub(3),Tr0)
assign(U_n.sub(4),Dn0)
assign(U_n.sub(5),D0)
assign(U_n.sub(6),Mn0)
assign(U_n.sub(7),M0)
assign(U_n.sub(8),C0)
assign(U_n.sub(9),N0)
assign(U_n.sub(10),A0)
assign(U_n.sub(11),H0)
assign(U_n.sub(12),IL120)
assign(U_n.sub(13),IL100)
assign(U_n.sub(14),IL60)

Tn_n,Th_n,Tc_n,Tr_n,Dn_n,D_n,Mn_n,M_n,C_n,N_n,A_n,H_n,IL12_n,IL10_n,IL6_n= U_n.split()
##############################################################


###############################################################
#Sum of RHS
###############################################################
def RHS_sum(U,p):
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14= U_n.split()
    RHS_Tn = p[54]-(p[0]*x11+p[1]*x5+p[2]*x12)*x0-(p[3]*x5+p[4]*x12)*x0-(p[5]*x5+p[40])*x0
    RHS_Th = (p[0]*x11+p[1]*x5+p[2]*x12)*x0-(p[33]*x3+p[34]*x13+p[35])*x1
    RHS_Tc = (p[3]*x5+p[4]*x12)*x0-(p[36]*x3+p[37]*x13+p[38])*x2
    RHS_Tr = (p[5]*x5)*x0-p[39]*x3
    RHS_Dn = p[55]-(p[6]*x8+p[7]*x11)*x4-p[43]*x4
    RHS_D = (p[6]*x8+p[7]*x11)*x4-(p[41]*x8+p[42])*x5
    RHS_Mn = p[56]-(p[8]*x13+p[9]*x12+p[10]*x1+p[45])*x6
    RHS_M = (p[8]*x13+p[9]*x12+p[10]*x1)*x6-p[44]*x7
    RHS_C = (p[11]+p[12]*x14+p[13]*x10)*(1-x8/p[58])*x8-(p[46]*x2+p[47])*x8
    RHS_N = p[57]*(p[46]*x2+p[47])*x8-p[49]*x9
    RHS_A = (p[14]*x10)*(1-x10/p[59])-p[48]*x10
    return RHS_Th+RHS_Tc+RHS_Tr+RHS_Dn+RHS_D+RHS_M+RHS_C+RHS_N+RHS_A
RHS = RHS_sum(U_n,params)
RHS_MECH_ = project(RHS,S1)
##############################################################


#######################################################################
#Mesh and remeshing related info and
#######################################################################
numCells = mesh.num_cells()
mesh.smooth(100)
Counter=0
#######################################################################

#######################################################################
#Save the last step cancer dynamics to make sure our dynamics match.
#######################################################################
vtkfile = File('output/C.pvd')
#######################################################################

#######################################################################
#Since displacement has already been calculated we can take bigger time steps
#This adjoint process takes up a lot of memory. So the bigger the jump the less waste of memory
#######################################################################
for n in range(num_steps/4):

     if j>=1:
         ##############################################################
         #Loop info update and printing
         ##############################################################
         print(t,flush=True)
         t+=dt
         ##############################################################

         ##############################################################
         #Create displacement for mesh movement from precalculated displacements
         ##############################################################
         u__ = Function(VV1,"displacement/u%d.xml" %(t))
         dis = k*u__
         displ = project(dis,VV1)
         ALE.move(mesh,displ)
         #############################################################

         ##############################################################


         ##############################################################
         #Update biology PDE and solve
         #############################################################
         F1 = ((Tn-Tn_n)/k)*v1*dx-(params[54]-(params[0]*H+params[1]*D+params[2]*IL12)*Tn-(params[3]*D+params[4]*IL12)*Tn-(params[5]*D+params[40])*Tn)*v1*dx\
         + ((Th-Th_n)/k)*v2*dx+D_Th*dot(grad(Th),grad(v2))*dx+coeff*Th*div(u__)*v2*dx-((params[0]*H+params[1]*D+params[2]*IL12)*Tn-(params[33]*Tr+params[34]*IL10+params[35])*Th)*v2*dx+kappaTh*(Th-Th_R)*v2*ds(1)\
         + ((Tc-Tc_n)/k)*v3*dx+D_Tc*dot(grad(Tc),grad(v3))*dx+coeff*Tc*div(u__)*v3*dx-((params[3]*D+params[4]*IL12)*Tn-(params[36]*Tr+params[37]*IL10+params[38])*Tc)*v3*dx+kappaTc*(Tc-Tc_R)*v3*ds(1)\
         + ((Tr-Tr_n)/k)*v4*dx+D_Tr*dot(grad(Tr),grad(v4))*dx+coeff*Tr*div(u__)*v4*dx-((params[5]*D)*Tn-params[39]*Tr)*v4*dx+kappaTr*(Tr-Tr_R)*v4*ds(1)\
         + ((Dn-Dn_n)/k)*v5*dx+D_Dn*dot(grad(Dn),grad(v5))*dx+coeff*Dn*div(u__)*v5*dx-(params[55]-(params[6]*C+params[7]*H)*Dn-params[43]*Dn)*v5*dx+kappaDn*(Dn-Dn_R)*v5*ds(1)\
         + ((D-D_n)/k)*v6*dx+D_D*dot(grad(D),grad(v6))*dx+coeff*D*div(u__)*v6*dx-((params[6]*C+params[7]*H)*Dn-(params[41]*C+params[42])*D)*v6*dx\
         + ((Mn-Mn_n)/k)*v7*dx-(params[56]-(params[8]*IL10+params[9]*IL12+params[10]*Th+params[45])*Mn)*v7*dx\
         + ((M-M_n)/k)*v8*dx+D_M*dot(grad(M),grad(v8))*dx+coeff*M*div(u__)*v8*dx-((params[8]*IL10+params[9]*IL12+params[10]*Th)*Mn-params[44]*M)*v8*dx+kappaM*(M-M_R)*v8*ds(1)\
         + ((C-C_n)/k)*v9*dx+D_C*dot(grad(C),grad(v9))*dx+coeff*C*div(u__)*v9*dx-((params[11]+parameter1*IL6+parameter2*A)*(1-C/params[58])*C-(params[46]*Tc+params[47])*C)*v9*dx\
         + ((N-N_n)/k)*v10*dx+D_N*dot(grad(N),grad(v10))*dx+coeff*N*div(u__)*v10*dx-(params[57]*(params[46]*Tc+params[47])*C-params[49]*N)*v10*dx\
         + ((A-A_n)/k)*v11*dx+D_A*dot(grad(A),grad(v11))*dx+coeff*A*div(u__)*v11*dx-((params[14]*A)*(1-A/params[59])-params[48]*A)*v11*dx\
         + ((H-H_n)/k)*v12*dx+D_H*dot(grad(H),grad(v12))*dx-(params[15]*D+params[16]*N+params[17]*M+params[18]*Tc+params[19]*C-params[50]*H)*v12*dx\
         + ((IL12-IL12_n)/k)*v13*dx+D_IL12*dot(grad(IL12),grad(v13))*dx-(params[20]*M+params[21]*D+params[22]*Th+params[23]*Tc-params[51]*IL12)*v13*dx\
         + ((IL10-IL10_n)/k)*v14*dx+D_IL10*dot(grad(IL10),grad(v14))*dx-(params[24]*M+params[25]*D+params[26]*Tr+params[27]*Th+params[28]*Tc+params[29]*C-params[52]*IL10)*v14*dx\
         + ((IL6-IL6_n)/k)*v15*dx+D_IL6*dot(grad(IL6),grad(v15))*dx-(params[30]*A+params[31]*M+params[32]*D-params[53]*IL6)*v15*dx

         bc = []
         solve(F1==0,U,bc)


         ##############################################################
         #Making the negative values caused by coarse projection zero
         #For smooth enough IC is not needed. So you can comment or leave like this
         ##############################################################
         i_Th=np.argwhere(U.sub(1).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(1).vector()[i_Th[:,0]] = 1.e-16

         i_Tc=np.argwhere(U.sub(2).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(2).vector()[i_Tc[:,0]] = 1.e-16

         i_Tr=np.argwhere(U.sub(3).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(3).vector()[i_Tr[:,0]] = 1.e-16

         i_Dn=np.argwhere(U.sub(4).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(4).vector()[i_Dn[:,0]] = 1.e-16

         i_D=np.argwhere(U.sub(5).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(5).vector()[i_D[:,0]] = 1.e-16

         i_M=np.argwhere(U.sub(7).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(7).vector()[i_M[:,0]] = 1.e-16

         i_C=np.argwhere(U.sub(8).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(8).vector()[i_C[:,0]] = 1.e-16

         i_N=np.argwhere(U.sub(9).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(9).vector()[i_N[:,0]] = 1.e-16

         i_A=np.argwhere(U.sub(10).vector().get_local()[:]<=0)  #making negatives zero
         U.sub(10).vector()[i_A[:,0]] = 1.e-16
         ##############################################################


         ##############################################################
         #Saving info of the previous time step
         ##############################################################
         U_n.assign(U)
         Tn_n,Th_n,Tc_n,Tr_n,Dn_n,D_n,Mn_n,M_n,C_n,N_n,A_n,H_n,IL12_n,IL10_n,IL6_n= U_n.split()
         ##############################################################

         #######################################################################



     ##############################################################
     #Update loop info
     ##############################################################
     j+=1
     ##############################################################



##############################################################
#Save the last cancer dynamics
vtkfile<<(C_n,t)
##############################################################

##############################################################
#Sensitivity Analysis using adjoint method
##############################################################

##############################################################
#Define your functional
##############################################################
J1 = assemble(C_n*dx)
##############################################################


##############################################################
#Defining control parameters
##############################################################

DTh_ct = Control(D_Th)
DTc_ct = Control(D_Tc)
DTr_ct = Control(D_Tr)
DDn_ct = Control(D_Dn)
DD_ct = Control(D_D)
DM_ct = Control(D_M)
DC_ct = Control(D_C)
DN_ct = Control(D_N)
DA_ct = Control(D_A)
DH_ct = Control(D_H)
DIL12_ct = Control(D_IL12)
DIL10_ct = Control(D_IL10)
DIL6_ct = Control(D_IL6)
kappaTh_ct = Control(kappaTh)
kappaTc_ct = Control(kappaTc)
kappaTr_ct = Control(kappaTr)
kappaDn_ct = Control(kappaDn)
kappaM_ct = Control(kappaM)
Th_R_ct = Control(Th_R)
Tc_R_ct = Control(Tc_R)
Tr_R_ct = Control(Tr_R)
Dn_R_ct = Control(Dn_R)
M_R_ct = Control(M_R)
##############################################################
print('Assemble and Controls done!')
##############################################################

##############################################################
#The rest is calculating gradient with respect to control PDE parameters and saving them as sensitivities
#The generated output is in the order of parameters given in line 182 or the input file parameters.csv
##############################################################
SS = []

dJ1dDTh = compute_gradient(J1, DTh_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTh,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('1st Gradient done!')
##############################################################
dJ1dDTc = compute_gradient(J1, DTc_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTc,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('2nd Gradient done!')
##############################################################
dJ1dDTr = compute_gradient(J1, DTr_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTr,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('3rd Gradient done!')
##############################################################
dJ1dDDn = compute_gradient(J1, DDn_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDDn,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('4th Gradient done!')
##############################################################
dJ1dDD = compute_gradient(J1, DD_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDD,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('5th Gradient done!')
##############################################################
dJ1dDM = compute_gradient(J1, DM_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDM,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('6th Gradient done!')
##############################################################
dJ1dDC = compute_gradient(J1, DC_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDC,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('7th Gradient done!')
##############################################################
dJ1dDN = compute_gradient(J1, DN_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDDn,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('8th Gradient done!')
##############################################################
dJ1dDA = compute_gradient(J1, DA_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDA,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('8th Gradient done!')
##############################################################
dJ1dDH = compute_gradient(J1, DH_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDH,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('9th Gradient done!')
##############################################################
dJ1dDIL12 = compute_gradient(J1, DIL12_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDIL12,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('10th Gradient done!')
##############################################################
dJ1dDIL10 = compute_gradient(J1, DIL10_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDIL10,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('11th Gradient done!')
##############################################################
dJ1dDIL6 = compute_gradient(J1, DIL6_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDIL6,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('12th Gradient done!')
##############################################################
dJ1dkappaTh = compute_gradient(J1, kappaTh_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dkappaTh,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('13th Gradient done!')
##############################################################
dJ1dkappaTc = compute_gradient(J1, kappaTc_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dkappaTc,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('14th Gradient done!')
##############################################################
dJ1dkappaTr = compute_gradient(J1, kappaTr_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dkappaTr,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('15th Gradient done!')
##############################################################
dJ1dkappaDn = compute_gradient(J1, kappaDn_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dkappaDn,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('16th Gradient done!')
##############################################################
dJ1dkappaM = compute_gradient(J1, kappaM_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dkappaM,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('17th Gradient done!')
##############################################################
dJ1dTh_R = compute_gradient(J1, Th_R_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dTh_R,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('18th Gradient done!')
##############################################################
dJ1dTc_R = compute_gradient(J1, Tc_R_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dTc_R,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('19th Gradient done!')
##############################################################
dJ1dTr_R = compute_gradient(J1, Tr_R_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dTr_R,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('20th Gradient done!')
##############################################################
dJ1dDn_R = compute_gradient(J1, Dn_R_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDn_R,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('21th Gradient done!')
##############################################################
dJ1dM_R = compute_gradient(J1, M_R_ct, options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dM_R,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('22th Gradient done!')
##############################################################
