from dolfin import *
import numpy as np

def reference_mapping(U_n,Total_Cells_1,S1,t,vtkfile):

    Tn_n,Th_n,Tc_n,Tr_n,Dn_n,D_n,Mn_n,M_n,C_n,N_n,A_n,H_n,IL12_n,IL10_n,IL6_n= U_n.split()

    Th1 = project(Th_n,S1)
    Tc1 = project(Tc_n,S1)
    Tr1 = project(Tr_n,S1)
    Dn1 = project(Dn_n,S1)
    D1 = project(D_n,S1)
    M1 = project(M_n,S1)
    C1 = project(C_n,S1)
    N1 = project(N_n,S1)
    A1 = project(A_n,S1)


    mesh1 = S1.mesh()
    ###############################################################
    mesh2= Mesh()
    xdmf = XDMFFile(mesh2.mpi_comm(), "Mesh.xdmf")
    xdmf.read(mesh2)
    mvc = MeshValueCollection("size_t", mesh2, 2)
    with XDMFFile("Mesh.xdmf") as infile:
        infile.read(mvc, "f")
    Volume = cpp.mesh.MeshFunctionSizet(mesh2, mvc)
    xdmf.close()
    mvc2 = MeshValueCollection("size_t", mesh2, 1)
    with XDMFFile("boundaries.xdmf") as infile:
        infile.read(mvc2, "f")
    bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh2, mvc2)
    ###############################################################

    S2 = FunctionSpace(mesh2,'P',1)
    R = FunctionSpace(mesh2,'R',0)

    #Defining dof to vertex and vertex to dof maps for deformed and reference regions
    dof2v_S1 = np.array(dof_to_vertex_map(S1), dtype=int)
    dof2v_S2 = np.array(dof_to_vertex_map(S2), dtype=int)
    v2dof_S1 = np.array(vertex_to_dof_map(S1), dtype=int)
    v2dof_S2 = np.array(vertex_to_dof_map(S2), dtype=int)
    #######################################################################


    Th2 = Function(S2)
    Th2Array = Th2.vector().get_local()
    Th2Array[v2dof_S2[dof2v_S1]] = Th1.vector().get_local()
    Th2.vector()[:]=Th2Array
    #######################################################################
    Tc2 = Function(S2)
    Tc2Array = Tc2.vector().get_local()
    Tc2Array[v2dof_S2[dof2v_S1]] = Tc1.vector().get_local()
    Tc2.vector()[:]=Tc2Array
    #######################################################################
    Tr2 = Function(S2)
    Tr2Array = Tr2.vector().get_local()
    Tr2Array[v2dof_S2[dof2v_S1]] = Tr1.vector().get_local()
    Tr2.vector()[:]=Tr2Array
    #######################################################################
    Dn2 = Function(S2)
    Dn2Array = Dn2.vector().get_local()
    Dn2Array[v2dof_S2[dof2v_S1]] = Dn1.vector().get_local()
    Dn2.vector()[:]=Dn2Array
    #######################################################################
    D2 = Function(S2)
    D2Array = D2.vector().get_local()
    D2Array[v2dof_S2[dof2v_S1]] = D1.vector().get_local()
    D2.vector()[:]=D2Array
    #######################################################################
    M2 = Function(S2)
    M2Array = M2.vector().get_local()
    M2Array[v2dof_S2[dof2v_S1]] = M1.vector().get_local()
    M2.vector()[:]=M2Array
    #######################################################################
    C2 = Function(S2)
    C2Array = C2.vector().get_local()
    C2Array[v2dof_S2[dof2v_S1]] = C1.vector().get_local()
    C2.vector()[:]=C2Array
    #######################################################################
    N2 = Function(S2)
    N2Array = N2.vector().get_local()
    N2Array[v2dof_S2[dof2v_S1]] = N1.vector().get_local()
    N2.vector()[:]=N2Array
    #######################################################################
    A2 = Function(S2)
    A2Array = A2.vector().get_local()
    A2Array[v2dof_S2[dof2v_S1]] = A1.vector().get_local()
    A2.vector()[:]=A2Array
    #######################################################################
    Total_Cells_2 = Function(S2)
    Total_Cells_2Array = Total_Cells_2.vector().get_local()
    Total_Cells_2Array[v2dof_S2[dof2v_S1]] = Total_Cells_1.vector().get_local()
    Total_Cells_2.vector()[:]=Total_Cells_2Array
    #######################################################################



    Th_total = assemble(Th2*dx)
    Th_total_ = project(Th_total,R)
    Tc_total = assemble(Tc2*dx)
    Tc_total_ = project(Tc_total,R)
    Tr_total = assemble(Tr2*dx)
    Tr_total_ = project(Tr_total,R)
    Dn_total = assemble(Dn2*dx)
    Dn_total_ = project(Dn_total,R)
    D_total = assemble(D2*dx)
    D_total_ = project(D_total,R)
    M_total = assemble(M2*dx)
    M_total_ = project(M_total,R)
    C_total = assemble(C2*dx)
    C_total_ = project(C_total,R)
    N_total = assemble(N2*dx)
    N_total_ = project(N_total,R)
    A_total = assemble(A2*dx)
    A_total_ = project(A_total,R)
    tot_total = assemble(Total_Cells_2*dx)
    tot_total_ = project(tot_total,R)


    Th2.rename('Th2','Th2')
    Tc2.rename('Tc2','Tc2')
    Tr2.rename('Tr2','Tr2')
    Dn2.rename('Dn2','Dn2')
    D2.rename('D2','D2')
    M2.rename('M2','M2')
    C2.rename('C2','C2')
    N2.rename('N2','N2')
    A2.rename('A2','A2')
    Total_Cells_2.rename('Total_Cells_2','Total_Cells_2')

    vtkfile[0].write(Th2,t)
    vtkfile[1].write(Tc2,t)
    vtkfile[2].write(Tr2,t)
    vtkfile[3].write(Dn2,t)
    vtkfile[4].write(D2,t)
    vtkfile[5].write(M2,t)
    vtkfile[6].write(C2,t)
    vtkfile[7].write(N2,t)
    vtkfile[8].write(A2,t)
    vtkfile[9].write(Total_Cells_2,t)


    Th_total_.rename('Th_total_','Th_total_')
    Tc_total_.rename('Tc_total_','Tc_total_')
    Tr_total_.rename('Tr_total_','Tr_total_')
    Dn_total_.rename('Dn_total_','Dn_total_')
    D_total_.rename('D_total_','D_total_')
    M_total_.rename('M_total_','M_total_')
    C_total_.rename('C_total_','C_total_')
    N_total_.rename('N_total_','N_total_')
    A_total_.rename('A_total_','A_total_')
    tot_total_.rename('tot_total_','tot_total_')

    vtkfile[10].write(Th_total_,t)
    vtkfile[11].write(Tc_total_,t)
    vtkfile[12].write(Tr_total_,t)
    vtkfile[13].write(Dn_total_,t)
    vtkfile[14].write(D_total_,t)
    vtkfile[15].write(M_total_,t)
    vtkfile[16].write(C_total_,t)
    vtkfile[17].write(N_total_,t)
    vtkfile[18].write(A_total_,t)
    vtkfile[19].write(tot_total_,t)
    ##############################################################
