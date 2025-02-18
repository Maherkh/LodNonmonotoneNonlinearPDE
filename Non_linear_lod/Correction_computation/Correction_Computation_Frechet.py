 
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from Non_linear_lod.Patch_info.patch_restriction_matrices import Patch_matrices
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights
import ray

@ray.remote
def Quara(ind,size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro, St,Ma,Co,B, Ch):
    CorSize=(N+1)**2
    FinSize=len(F_Nod[:,0])
    Qh = lil_matrix((CorSize, FinSize))   
    # We do the computations for each element in the coarse mesh:
    for l in [ind]:   
        RlH, Rlh, Nlh =Patch_matrices(l, size,C_ele, C_Nod,F_Nod, C_bnd,F_bnd )
        NlH = RlH.shape[0]  # Local coarse space size
        StiffPatch = Rlh @ (St+Co) @ Rlh.T  # Fine-scale stiffness matrix for the patch
        Cl = RlH @ Ch @ Rlh.T  # Local constraint matrix, size NlH * Nlh
        TH = np.zeros((4, CorSize))  # Local to global mapping
        TH[np.arange(4), C_ele[l,:]]=1
        # Initialize the patch stiffness matrix as a sparse matrix
        PatchStiff = lil_matrix((FinSize, FinSize))  # Global fine stiffness for the patch
        Patch_fine_element = set()
        # Identify which fine elements belong to the patch
        for i in range(F_ele.shape[0]): 
            element_nodes = F_Nod[F_ele[i, :]] 
            midpoint = np.mean(element_nodes, axis=0)
            if (C_Nod[C_ele[l, 0]][0] < midpoint[0] < C_Nod[C_ele[l, 3]][0] and
                C_Nod[C_ele[l, 0]][1] < midpoint[1] < C_Nod[C_ele[l, 3]][1]):
                Patch_fine_element.add(tuple(F_ele[i]))
        Patch_fine_element = np.array(list(Patch_fine_element))
        # Loop over fine elements in the patch
        for q in range(Patch_fine_element.shape[0]):  
            ele_coords = F_Nod[Patch_fine_element[q, :], :]
            element=Patch_fine_element[q,:]
            u_star_elem=u_str[element]
            m_point=np.mean(ele_coords, axis=0)
            m_u=np.mean(u_star_elem, axis=0)
            A_star=Anon(np.array([(m_point[0],m_point[1])]),np.array([m_u]))
            A_u=A_du(np.array([(m_point[0],m_point[1])]),np.array([m_u]))
            for gp, w1 in zip(gauss_points, weights):
                xi, yi = gp
                Phi_val, grad_val_ref = ref_basis(xi, yi)
                J = np.dot(grad_val_ref.T, ele_coords)  
                 
                detJ= J[0,0]*J[1,1]-J[0,1]*J[1,0]
                invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

                grad_val_phys = np.dot(invJ, grad_val_ref.T).T  # Shape: (4, 2)
                grad_u_star_ref = np.dot(grad_val_ref.T, u_str[element])  # Gradient of u* on reference element
                grad_u_star_phys = np.dot(invJ.T, grad_u_star_ref)  # Gradient of u* in physical space
                for i in range(4):  #Loop over test functions (rows of K_local)
                    for j in range(4):  # Loop over trial functions (columns of K_local)
                         
                        grad_Phi_j_phys = grad_val_phys[j, :]  # Grad(Phi_j) in physical space
                        convective_term =  A_u[0] * grad_u_star_phys  # A_u * grad u*
                        PatchStiff[element[i],element[j]]+=Phi_val[i] * np.dot(convective_term, grad_Phi_j_phys) * detJ * w1 + A_star[0]*np.dot(grad_val_phys[i,:], grad_val_phys[j,:]) * detJ * w1

        StiffPatch = Rlh @ (St+Co) @ Rlh.T   
        StiffPatch = csc_matrix(StiffPatch)   
        lu = splu(StiffPatch)  # Sparse LU decomposition
        # Compute preconditioned residual
        rl = -TH @ B @ pro @ PatchStiff @ Rlh.T
        
        Yl = lu.solve((Cl.toarray()).T)
        # Schur complement inverse
        S=csc_matrix(Cl @ Yl)
        lu1 = splu(S)

        # Compute the correction for each coarse space function that has support on the element
        cd = TH.shape[0]  
        wl = np.zeros((cd, Nlh))
        for k in range(cd):
            if np.any(rl[k, :]):
                qk = lu.solve( rl[k, :].T)
                lambdak = lu1.solve( Cl @ qk)
                wl[k, :] = qk - Yl @ lambdak
        Qh += TH.T @ wl @ Rlh
    return csr_matrix(Qh)