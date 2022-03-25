'''
A Bio-Mechanical PDE model of breast tumor progression in MMTV-PyMT mice

**Calculates sensitivities of total cancer cell to reaction Parameters**

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

###############################################################
#ODE parameters taken as FEniCS Constant object for sensitivity purposes
###############################################################
params0 = Constant(params[0])
params1 = Constant(params[1])
params2 = Constant(params[2])
params3 = Constant(params[3])
params4 = Constant(params[4])
params5 = Constant(params[5])
params6 = Constant(params[6])
params7 = Constant(params[7])
params8 = Constant(params[8])
params9 = Constant(params[9])
params10 = Constant(params[10])
params11 = Constant(params[11])
params12 = Constant(params[12])
params13 = Constant(params[13])
params14 = Constant(params[14])
params15 = Constant(params[15])
params16 = Constant(params[16])
params17 = Constant(params[17])
params18 = Constant(params[18])
params19 = Constant(params[19])
params20 = Constant(params[20])
params21 = Constant(params[21])
params22 = Constant(params[22])
params23 = Constant(params[23])
params24 = Constant(params[24])
params25 = Constant(params[25])
params26 = Constant(params[26])
params27 = Constant(params[27])
params28 = Constant(params[28])
params29 = Constant(params[29])
params30 = Constant(params[30])
params31 = Constant(params[31])
params32 = Constant(params[32])
params33 = Constant(params[33])
params34 = Constant(params[34])
params35 = Constant(params[35])
params36 = Constant(params[36])
params37 = Constant(params[37])
params38 = Constant(params[38])
params39 = Constant(params[39])
params40 = Constant(params[40])
params41 = Constant(params[41])
params42 = Constant(params[42])
params43 = Constant(params[43])
params44 = Constant(params[44])
params45 = Constant(params[45])
params46 = Constant(params[46])
params47 = Constant(params[47])
params48 = Constant(params[48])
params49 = Constant(params[49])
params50 = Constant(params[50])
params51 = Constant(params[51])
params52 = Constant(params[52])
params53 = Constant(params[53])
params54 = Constant(params[54])
params55 = Constant(params[55])
params56 = Constant(params[56])
params57 = Constant(params[57])
params58 = Constant(params[58])
params59 = Constant(params[59])
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
         #constructing mechanical problem based on updated RHS, crvt and Normals

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
         F1 = ((Tn-Tn_n)/k)*v1*dx-(params54-(params0*H+params1*D+params2*IL12)*Tn-(params3*D+params4*IL12)*Tn-(params5*D+params40)*Tn)*v1*dx\
         + ((Th-Th_n)/k)*v2*dx+D_Th*dot(grad(Th),grad(v2))*dx+coeff*Th*div(u__)*v2*dx-((params0*H+params1*D+params2*IL12)*Tn-(params33*Tr+params34*IL10+params35)*Th)*v2*dx+kappaTh*(Th-Th_R)*v2*ds(1)\
         + ((Tc-Tc_n)/k)*v3*dx+D_Tc*dot(grad(Tc),grad(v3))*dx+coeff*Tc*div(u__)*v3*dx-((params3*D+params4*IL12)*Tn-(params36*Tr+params37*IL10+params38)*Tc)*v3*dx+kappaTc*(Tc-Tc_R)*v3*ds(1)\
         + ((Tr-Tr_n)/k)*v4*dx+D_Tr*dot(grad(Tr),grad(v4))*dx+coeff*Tr*div(u__)*v4*dx-((params5*D)*Tn-params39*Tr)*v4*dx+kappaTr*(Tr-Tr_R)*v4*ds(1)\
         + ((Dn-Dn_n)/k)*v5*dx+D_Dn*dot(grad(Dn),grad(v5))*dx+coeff*Dn*div(u__)*v5*dx-(params55-(params6*C+params7*H)*Dn-params43*Dn)*v5*dx+kappaDn*(Dn-Dn_R)*v5*ds(1)\
         + ((D-D_n)/k)*v6*dx+D_D*dot(grad(D),grad(v6))*dx+coeff*D*div(u__)*v6*dx-((params6*C+params7*H)*Dn-(params41*C+params42)*D)*v6*dx\
         + ((Mn-Mn_n)/k)*v7*dx-(params56-(params8*IL10+params9*IL12+params10*Th+params45)*Mn)*v7*dx\
         + ((M-M_n)/k)*v8*dx+D_M*dot(grad(M),grad(v8))*dx+coeff*M*div(u__)*v8*dx-((params8*IL10+params9*IL12+params10*Th)*Mn-params44*M)*v8*dx+kappaM*(M-M_R)*v8*ds(1)\
         + ((C-C_n)/k)*v9*dx+D_C*dot(grad(C),grad(v9))*dx+coeff*C*div(u__)*v9*dx-((params11+params12*IL6+params13*A)*(1-C/params58)*C-(params46*Tc+params47)*C)*v9*dx\
         + ((N-N_n)/k)*v10*dx+D_N*dot(grad(N),grad(v10))*dx+coeff*N*div(u__)*v10*dx-(params57*(params46*Tc+params47)*C-params49*N)*v10*dx\
         + ((A-A_n)/k)*v11*dx+D_A*dot(grad(A),grad(v11))*dx+coeff*A*div(u__)*v11*dx-((params14*A)*(1-A/params59)-params48*A)*v11*dx\
         + ((H-H_n)/k)*v12*dx+D_H*dot(grad(H),grad(v12))*dx-(params15*D+params16*N+params17*M+params18*Tc+params19*C-params50*H)*v12*dx\
         + ((IL12-IL12_n)/k)*v13*dx+D_IL12*dot(grad(IL12),grad(v13))*dx-(params20*M+params21*D+params22*Th+params23*Tc-params51*IL12)*v13*dx\
         + ((IL10-IL10_n)/k)*v14*dx+D_IL10*dot(grad(IL10),grad(v14))*dx-(params24*M+params25*D+params26*Tr+params27*Th+params28*Tc+params29*C-params52*IL10)*v14*dx\
         + ((IL6-IL6_n)/k)*v15*dx+D_IL6*dot(grad(IL6),grad(v15))*dx-(params30*A+params31*M+params32*D-params53*IL6)*v15*dx

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

     ##############################################################


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
print('Assemble and Controls done!')
##############################################################

##############################################################
#The rest is calculating gradient with respect to control ODE parameters and saving them as sensitivities
#The generated output is in the order of parameters given in line 182 or the input file parameters.csv
##############################################################

SS = []

dJ1dp0 = compute_gradient(J1, Control(params0), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp0,S1).vector()[:]))
print('1 Gradient done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp1 = compute_gradient(J1, Control(params1), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp1,S1).vector()[:]))
print('2 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp2 = compute_gradient(J1, Control(params2), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp2,S1).vector()[:]))
print('3 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp3 = compute_gradient(J1, Control(params3), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp3,S1).vector()[:]))
print('4 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp4 = compute_gradient(J1, Control(params4), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp4,S1).vector()[:]))
print('5 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp5 = compute_gradient(J1, Control(params5), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp5,S1).vector()[:]))
print('6 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp6 = compute_gradient(J1, Control(params6), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp6,S1).vector()[:]))
print('7 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp7 = compute_gradient(J1, Control(params7), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp7,S1).vector()[:]))
print('8 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp8 = compute_gradient(J1, Control(params8), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp8,S1).vector()[:]))
print('9 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp9 = compute_gradient(J1, Control(params9), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp9,S1).vector()[:]))
print('10 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp10 = compute_gradient(J1, Control(params10), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp10,S1).vector()[:]))
print('11 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp11 = compute_gradient(J1, Control(params11), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp11,S1).vector()[:]))
print('12 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp12 = compute_gradient(J1, Control(params12), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp12,S1).vector()[:]))
print('13 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp13 = compute_gradient(J1, Control(params13), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp13,S1).vector()[:]))
print('14 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp14 = compute_gradient(J1, Control(params14), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp14,S1).vector()[:]))
print('15 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp15 = compute_gradient(J1, Control(params15), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp15,S1).vector()[:]))
print('16 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp16 = compute_gradient(J1, Control(params16), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp16,S1).vector()[:]))
print('17 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp17 = compute_gradient(J1, Control(params17), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp17,S1).vector()[:]))
print('18 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp18 = compute_gradient(J1, Control(params18), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp18,S1).vector()[:]))
print('19 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp19 = compute_gradient(J1, Control(params19), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp19,S1).vector()[:]))
print('20 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp20 = compute_gradient(J1, Control(params20), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp20,S1).vector()[:]))
print('21 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp21 = compute_gradient(J1, Control(params21), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp21,S1).vector()[:]))
print('22 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp22 = compute_gradient(J1, Control(params22), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp22,S1).vector()[:]))
print('23 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp23 = compute_gradient(J1, Control(params23), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp23,S1).vector()[:]))
print('24 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp24 = compute_gradient(J1, Control(params24), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp24,S1).vector()[:]))
print('25 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp25 = compute_gradient(J1, Control(params25), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp25,S1).vector()[:]))
print('26 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp26 = compute_gradient(J1, Control(params26), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp26,S1).vector()[:]))
print('27 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp27 = compute_gradient(J1, Control(params27), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp27,S1).vector()[:]))
print('28 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp28 = compute_gradient(J1, Control(params28), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp28,S1).vector()[:]))
print('29 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp29 = compute_gradient(J1, Control(params29), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp29,S1).vector()[:]))
print('30 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp30 = compute_gradient(J1, Control(params30), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp30,S1).vector()[:]))
print('31 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp31 = compute_gradient(J1, Control(params31), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp31,S1).vector()[:]))
print('32 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp32 = compute_gradient(J1, Control(params32), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp32,S1).vector()[:]))
print('33 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp33 = compute_gradient(J1, Control(params33), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp33,S1).vector()[:]))
print('34 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp34 = compute_gradient(J1, Control(params34), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp34,S1).vector()[:]))
print('35 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp35 = compute_gradient(J1, Control(params35), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp35,S1).vector()[:]))
print('36 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp36 = compute_gradient(J1, Control(params36), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp36,S1).vector()[:]))
print('37 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp37 = compute_gradient(J1, Control(params37), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp37,S1).vector()[:]))
print('38 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp38 = compute_gradient(J1, Control(params38), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp38,S1).vector()[:]))
print('39 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp39 = compute_gradient(J1, Control(params39), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp39,S1).vector()[:]))
print('40 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp40 = compute_gradient(J1, Control(params40), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp40,S1).vector()[:]))
print('41 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp41 = compute_gradient(J1, Control(params41), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp41,S1).vector()[:]))
print('42 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp42 = compute_gradient(J1, Control(params42), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp42,S1).vector()[:]))
print('43 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp43 = compute_gradient(J1, Control(params43), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp43,S1).vector()[:]))
print('44 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp44 = compute_gradient(J1, Control(params44), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp44,S1).vector()[:]))
print('45 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp45 = compute_gradient(J1, Control(params45), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp45,S1).vector()[:]))
print('46 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp46 = compute_gradient(J1, Control(params46), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp46,S1).vector()[:]))
print('47 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp47 = compute_gradient(J1, Control(params47), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp47,S1).vector()[:]))
print('48 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp48 = compute_gradient(J1, Control(params48), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp48,S1).vector()[:]))
print('49 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp49 = compute_gradient(J1, Control(params49), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp49,S1).vector()[:]))
print('50 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp50= compute_gradient(J1, Control(params50), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp50,S1).vector()[:]))
print('51 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp51= compute_gradient(J1, Control(params51), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp51,S1).vector()[:]))
print('52 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp52= compute_gradient(J1, Control(params52), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp52,S1).vector()[:]))
print('53 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp53= compute_gradient(J1, Control(params53), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp53,S1).vector()[:]))
print('54 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp54= compute_gradient(J1, Control(params54), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp54,S1).vector()[:]))
print('55 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp55= compute_gradient(J1, Control(params55), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp55,S1).vector()[:]))
print('56 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp56= compute_gradient(J1, Control(params56), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp56,S1).vector()[:]))
print('57 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp57= compute_gradient(J1, Control(params57), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp57,S1).vector()[:]))
print('58 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp58= compute_gradient(J1, Control(params58), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp58,S1).vector()[:]))
print('59 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp59= compute_gradient(J1, Control(params59), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp59,S1).vector()[:]))
print('60 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
