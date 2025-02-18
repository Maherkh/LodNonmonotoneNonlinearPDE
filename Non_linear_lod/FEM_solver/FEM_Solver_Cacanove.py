import numpy as np
from Non_linear_lod.FEM_solver.FEM_Matrices import assemble_system
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import scipy.sparse as sparse
 
 

def solveFEM(u_str,F_Nod, F_ele, F_bnd,N,right_hand, Anonlin,maxiter,tol):
    Stiff, fh, Mass= assemble_system(F_Nod, F_ele, N,u_str, right_hand, Anonlin)
    u0ref = np.zeros((N+1)**2)
    uFullref = np.copy(u0ref)
    AFull = Stiff
    free_index=np.array([i for i in range(AFull.shape[0]) if i not in  F_bnd ])
    AH_reduced=AFull[np.ix_(free_index,free_index)]
    fH_reduced=fh[free_index]
    resref = np.linalg.norm((AFull[free_index][:,free_index] * uFullref[free_index] - fH_reduced))
    itref = 0
    while resref >tol  and itref < maxiter:
        #solve
        if itref==0:
            AFree = AH_reduced
            bFree = fH_reduced
            uFree = sparse.linalg.spsolve(AFree, bFree)
            uFullref[free_index] = uFree
        else:
            AFree = AFull_new[free_index][:, free_index]
            bFree = fH_reduced
            uFree = sparse.linalg.spsolve(AFree, bFree)
            uFullref[free_index] = uFree
        # compute FEM residual of nonlinear equation
        AFull_new, _, _ = assemble_system(F_Nod, F_ele, N,uFullref,right_hand,Anonlin)
        resref = np.linalg.norm((AFull_new[free_index][:,free_index] * uFullref[free_index] - fH_reduced))
        print('For the Reference Solution:residual in {}th iteration is {}'.format(itref, resref), end='\n', flush=True)
        itref += 1
    return uFullref