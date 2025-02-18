#FINITE ELEMTN METHOD
import numpy as np
from scipy.sparse.linalg import spsolve
# Here we creat the geometries of th problem, Fine and coarse meshes:
#from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from Non_linear_lod.Patch_info.patch_restriction_matrices import Patch_matrices
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights


import ray
@ray.remote


def Quara(ind,size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro,Stf,B,Ch):
    # The needed data for computation of global corrector matrix
    CorSize=(N+1)**2

    FinSize=len(F_Nod[:,0])
    # Global constraint matrix of size NH*Nh
    Qh = lil_matrix((CorSize, FinSize))  # Correction matrix

    #Computations of each  coarse  element:
    for l in [ind]:   
    #for l in range(C_ele.shape[0]):
        RlH, Rlh, Nlh = Patch_matrices(l, size,C_ele, C_Nod,F_Nod, C_bnd,F_bnd )
        StiffPatch = Rlh @ Stf @ Rlh.T   
        Cl = RlH @ Ch @ Rlh.T  # Local constraint matrix, size NlH * Nlh
        TH = np.zeros((4, CorSize))  # Local to global mapping
        TH[np.arange(4), C_ele[l, :]]=1

        PatchStiff = lil_matrix((FinSize, FinSize))    # Initialize the patch stiffness matrix
        Patch_fine_element = set()
    
        for i in range(F_ele.shape[0]):   
             
            element_nodes = F_Nod[F_ele[i, :]]   
            midpoint = np.mean(element_nodes, axis=0)
            if (C_Nod[C_ele[l, 0]][0] < midpoint[0] < C_Nod[C_ele[l, 3]][0] and
                C_Nod[C_ele[l, 0]][1] < midpoint[1] < C_Nod[C_ele[l, 3]][1]):
                Patch_fine_element.add(tuple(F_ele[i]))
        Patch_fine_element = np.array(list(Patch_fine_element))
        # Accumulate the local stiffness contribution into the global patch stiffness
        for i in range(Patch_fine_element.shape[0]):
            # Coordinates of the fine element nodes
            ele_coords = F_Nod[Patch_fine_element[i, :], :]
            element=Patch_fine_element[i,:]
            u_star_elem=u_str[element]
            m_point=np.mean(ele_coords, axis=0)
            m_u=np.mean(u_star_elem, axis=0)
            A_star=Anon(np.array([(m_point[0],m_point[1])]),np.array([m_u]))
            for gp, w1 in zip(gauss_points, weights):
                x, y = gp
                Phi_val, grad_val = ref_basis(x, y)
                J = np.dot(grad_val.T, ele_coords)  # Compute Jacobian and its inverse
                detJ = np.linalg.det(J)
                invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
                 
                grad_N = np.dot(invJ, grad_val.T)
                 
                for q in range(4):
                    for j in range(4):
                        PatchStiff[element[q],element[j]] += A_star[0]*np.dot(grad_N[:, q], grad_N[:, j]) * detJ * w1
        
        StiffPatch = Rlh @ Stf @ Rlh.T   
        StiffPatch = csc_matrix(StiffPatch)   
        lu = splu(StiffPatch)   
        rl = -TH @ B @ pro @ PatchStiff @ Rlh.T
        Yl=lu.solve((Cl.toarray()).T)

        # Schur complement inverse
        Sl=csc_matrix(Cl @ Yl)
        lu1=splu(Sl)
        # Compute the correction for each coarse space function that has support on the element
        cd = TH.shape[0]   
        wl = np.zeros((cd, Nlh))
        for k in range(cd):
            if np.any(rl[k, :]):
                qk = lu.solve( rl[k, :].T)
                lambdak = lu1.solve((Cl @ qk))
                wl[k, :] = qk - Yl @ lambdak
         
        Qh += TH.T @ wl @ Rlh
    return csr_matrix(Qh)