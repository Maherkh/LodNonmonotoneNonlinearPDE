
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, diags
from scipy.linalg import eig
import pickle
import os
import sys
from Non_linear_lod.FEM_solver.FEM_Matrices import assemble_system
from Non_linear_lod.Prolongtion_Assembly import prolongation_matrix
from Non_linear_lod.Grid_Construction import Grid
from Non_linear_lod.FEM_solver.FEM_Solver_Cacanove import solveFEM
from Non_linear_lod.Correction_computation.Correction_Computation_Cacanove import Quara
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights
from Non_linear_lod.Patch_info.Patch_Construction import find_patch
import ray

#_______________________________________________________________________________________________________________
#Error indicator for a given element

@ray.remote(num_cpus=2)
def Indic_computation(J,C_ele,C_Nod,Stiff_New,ele_matrix,c_matrix,Righ_eigen,A_uN2,A_uN1,x,y):
    T_Cor_j=C_Nod[C_ele[J,:],]
    xj0,yj0=T_Cor_j[0,:]
    xj1,yj1=T_Cor_j[3,:]
    inside_mask_j= (xj0 <= x) & (x <= xj1) & (yj0 <= y) & (y <= yj1)
    inside_indices_j = inside_mask_j
    X_inside=Stiff_New[inside_mask_j,:][:,inside_mask_j] 
    max_val = np.max(np.abs(A_uN2[inside_indices_j] - A_uN1[inside_indices_j]))
    sum=(ele_matrix+c_matrix).dot(X_inside).dot((ele_matrix+c_matrix).T)
    eigenvalue, _=eig(sum.toarray(),Righ_eigen.toarray())
    indicator =(max_val**2)*(np.max(np.real(eigenvalue)))
    return indicator


def Indic_computation_patch(patch,C_ele,C_Nod,Stiff_New,ele_matrix,c_matrix,Righ_eigen,A_uN2,A_uN1,x,y):
# The needed data for computation of global corrector matrix
    indicator_total=0
    # Submit tasks for each coarse element in parallel
    futures = [Indic_computation.remote(J,C_ele,C_Nod,Stiff_New,ele_matrix,c_matrix,Righ_eigen,A_uN2,A_uN1,x,y) for J in patch]
    # Collect results from each future and accumulate into the global matrix Qh
    results=ray.get(futures)
    for reslut in results:
        indicator_total+=reslut
    return indicator_total

#_________________________________________________________________________________________________

def Indicator(l,k, uN1,uN2,corr,pro,Stiff_New,C_ele,C_Nod, F_Nod,Anon):
    c_matrix=corr[C_ele[l,:],]
    Patch=find_patch(C_ele,l,k)
    ele_matrix=pro[C_ele[l,:],]
    T_Cor=C_Nod[C_ele[l,:],]
    A_uN1=Anon(F_Nod,uN1)
    A_uN2=Anon(F_Nod,uN2)
    Stif=Stiff_New.copy()
    x0,y0=T_Cor[0,:]
    x1,y1=T_Cor[3,:]
    x=F_Nod[:, 0]
    y=F_Nod[:, 1]
# Loop through the limited range of FineNodes between start_idx and end_idx
    inside_mask= (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)
    Stif_inside=Stif[inside_mask,:][:,inside_mask]
    ele_matrix_inside=ele_matrix[:,inside_mask]
    c_matrix_inside=c_matrix[:,inside_mask]
    Righ_eigen=(ele_matrix_inside).dot(Stif_inside).dot((ele_matrix_inside).T)
    indicator=Indic_computation_patch(Patch,C_ele,C_Nod,Stiff_New,ele_matrix_inside,c_matrix_inside,Righ_eigen,A_uN2,A_uN1,x,y)
    return  np.sqrt(indicator) 

#___________________________________________________________________________________________________

@ray.remote(num_cpus=2)

def big_indic(l,size, uN1,uN2,Q_updated,pro,Stiff_new,C_ele,C_Nod, F_Nod, Tol,Anon):
    ind=Indicator(l,size, uN1,uN2,Q_updated,pro,Stiff_new,C_ele,C_Nod, F_Nod,Anon)
    return l if ind>Tol else None
