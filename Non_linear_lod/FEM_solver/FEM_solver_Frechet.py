import numpy as np
from Non_linear_lod.FEM_solver.FEM_Matrices import assemble_system, Conv_grad
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import scipy.sparse as sparse

def solve_Fem(N, u_st,F_Nod,F_ele, F_bnd,right_hand,Anonlin, Anonlin_du,maxiter,tol):
    Convection1 , RHS_New =Conv_grad(N,F_ele, F_Nod,u_st , Anonlin_du)
    Stiff1, fh, _=  assemble_system(F_Nod, F_ele, N, u_st,right_hand, Anonlin)
    u0ref = np.zeros((N+1)**2)
    uFullref = np.copy(u0ref)
    AFull = Stiff1+Convection1
    free_index=np.array([i for i in range(AFull.shape[0]) if i not in  F_bnd ])
    AH_reduced=AFull[np.ix_(free_index,free_index)]
    fH_reduced=(fh+RHS_New)[free_index]
    resref = np.linalg.norm((AFull[free_index][:,free_index] * uFullref[free_index] - fH_reduced))
    itref = 0
    while resref > tol and itref < maxiter:
        #solve
        if itref==0:
            bFull = fh+RHS_New
            AFree = AH_reduced
            bFree = bFull[free_index]
            uFree = sparse.linalg.spsolve(AFree, bFree)
            uFullref[free_index] = uFree
        else:
            bFull = fh+RHS_New_nw
            AFree = (AFull_new+Convection_nw)[free_index][:, free_index]
            bFree = bFull[free_index]
            uFree = sparse.linalg.spsolve(AFree, bFree)
            uFullref[free_index] = uFree
        # compute FEM residual of nonlinear equation
        AFull_new, _, _ = assemble_system(F_Nod, F_ele, N,uFullref,right_hand, Anonlin)
        Convection_nw, RHS_New_nw=Conv_grad(N,F_ele, F_Nod,uFullref , Anonlin_du)
        fH_reduced=(fh+RHS_New_nw)[free_index]
        resref = np.linalg.norm(((AFull_new+Convection_nw)[free_index][:,free_index] * uFullref[free_index] - fH_reduced))
        print('For the Reference Solution:residual in {}th iteration is {}'.format(itref, resref), end='\n', flush=True)
        itref += 1
    return uFullref