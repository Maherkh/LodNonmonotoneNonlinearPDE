 #Clean AdaptiveIterative Newton
import numpy as np
 
 

from scipy.sparse import   diags 
from scipy.sparse.linalg import splu,eigsh, eigs
 
 
 
 
 
from Non_linear_lod.Reference_Basis import ref_basis 
from Non_linear_lod.Patch_info.Patch_Construction import find_patch
import ray
#ray.init(num_cpus=48, include_dashboard=False)


def grad_compuation(ele,Nod,u_st):
    node_counter = np.zeros(Nod.shape[0])
    grad_u_node=np.zeros((Nod.shape[0],2))
    _, grad_val_ref = ref_basis(0.0, 0.0)
    for e in range(ele.shape[0]):
 
        elem_nodes = ele[e]  
        ele_coords = Nod[elem_nodes, :]
        u_elem = u_st[elem_nodes]
        #  Reference basis gradients (constant for linear elements)  
        J = grad_val_ref.T @ ele_coords
        #  Gradient in physical coordinates  
        grad_u_ref = grad_val_ref.T @ u_elem # (2,)
 
        grad_u_phys=np.linalg.solve(J,grad_u_ref)
        #  Accumulate to nodes  
        for a in elem_nodes:
            grad_u_node[a] += grad_u_phys
            node_counter[a] += 1
 
    grad_u_node /= node_counter[:, None]
    return grad_u_node

#___________________________________________________________________________________________________________________________________


#Error indicator for a given element
 
@ray.remote(num_cpus=2)
def Indic_computation(J,C_ele,C_Nod,Stiff_New,ele_matrix,c_matrix,Righ_eigen,Righ_eigMass,Mass,A_uN2,A_uN1,Ad_uN2,Adu_uN1,grad_uN1,grad_uN2,x,y):

    T_Cor_j=C_Nod[C_ele[J,:],]

    xj0,yj0=T_Cor_j[0,:]
    xj1,yj1=T_Cor_j[3,:]
    inside_mask_j= (xj0 <= x) & (x <= xj1) & (yj0 <= y) & (y <= yj1)
 
    Stiff_inside=Stiff_New[inside_mask_j][:,inside_mask_j] 
    Mass_inside=Mass[inside_mask_j][:,inside_mask_j] 
    max_val = np.max(np.abs(A_uN2[inside_mask_j] - A_uN1[inside_mask_j])) 
    max_val2=np.max(np.abs((diags(Ad_uN2)@grad_uN2)[inside_mask_j] - (diags(Adu_uN1)@grad_uN1)[inside_mask_j]))
    SSum=(ele_matrix+c_matrix).tocsr()
    sum=(SSum)@Stiff_inside@(SSum.T)
    sum_Mass=(SSum)@Mass_inside@(SSum.T)

    eigV, _ =eigsh(sum,M=Righ_eigen.toarray(),  k=1, which='LA')
    eigV1, _ =eigsh(sum_Mass,M=Righ_eigMass.toarray(),  k=1, which='LA')

    indicator_first=(max_val**2)*eigV[0]
    indicator_second=(max_val2**2)*eigV1[0]
    indicator=indicator_first+indicator_second
    return indicator

#___________________________________________________________________________________________________________________________________________

#Indicator computation corresponding to the patch
def Indic_computation_patch(patch,C_ele,C_Nod,F_Nod,F_ele,Stiff_New,ele_matrix,c_matrix,Righ_eigen,Rih_eigMass, Mass, uN1,uN2,A_uN2,A_uN1,Ad_uN2,Adu_uN1,x,y):
# The needed data for computation of global corrector matrix
    indicator_total=0
    grad_uN11=grad_compuation(F_ele,F_Nod,uN1)
    grad_uN22=grad_compuation(F_ele,F_Nod,uN2)

    
    patch_list=list(patch)
    Batch=40
    results=[]
    for start in range(0,len(patch), Batch):
        batch=patch_list[start: start+Batch]
        futures = [Indic_computation.remote(J,C_ele,C_Nod,Stiff_New,ele_matrix,c_matrix,Righ_eigen,Rih_eigMass, Mass,A_uN2,A_uN1,Ad_uN2,Adu_uN1,grad_uN11, grad_uN22,x,y) for J in batch]

    results=ray.get(futures)
    for result in results: 
        indicator_total+=result
    return indicator_total

#_________________________________________________________________________________________________



def Indicator(l,k, uN1,uN2,corr,pro,Stiff_New,Mass,C_ele,C_Nod, F_Nod,F_ele,A_uN1,A_uN2,Adu_uN1,Adu_uN2):

    c_matrix=corr[C_ele[l,:],]
    #sum=np.zeros((4,4))
    Patch=find_patch(C_ele,l,k)
    ele_matrix=pro[C_ele[l,:],]
    T_Cor=C_Nod[C_ele[l,:],]

    x0,y0=T_Cor[0,:]
    x1,y1=T_Cor[3,:]
    x=F_Nod[:, 0]
    y=F_Nod[:, 1]
# Loop through the limited range of FineNodes between start_idx and end_idx
    inside_mask= (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1) 
    X_X_X_inside=Stiff_New[inside_mask][:,inside_mask]
    X_XMass_inside=Mass[inside_mask][:,inside_mask]

    ele_matrix_inside=ele_matrix[:,inside_mask]
    c_matrix_inside=c_matrix[:,inside_mask]
    
    Righ_eig=(ele_matrix_inside).dot(X_X_X_inside).dot((ele_matrix_inside).T)
    Righ_eigMas=(ele_matrix_inside).dot(X_XMass_inside).dot((ele_matrix_inside).T)

    indicator=Indic_computation_patch(Patch,C_ele,C_Nod,F_Nod,F_ele,Stiff_New,ele_matrix_inside,c_matrix_inside,Righ_eig,Righ_eigMas, Mass, uN1,uN2,A_uN2,A_uN1,Adu_uN2,Adu_uN1,x,y)
    return  np.sqrt(indicator) 


#___________________________________________________________________________________________________
#Here we output the elements of indicator bigger than Tol

@ray.remote(num_cpus=2)
def big_indic(l,size, uN1,uN2,Q_updated,pro,Stiff_new,Mass, C_ele,C_Nod, F_Nod,F_ele, Tol,A_uN1,A_uN2,Adu_uN1, Adu_uN2):

    ind=Indicator(l,size, uN1,uN2,Q_updated,pro,Stiff_new,Mass, C_ele,C_Nod, F_Nod,F_ele,A_uN1,A_uN2, Adu_uN1,Adu_uN2)
    return l if ind>Tol else None
