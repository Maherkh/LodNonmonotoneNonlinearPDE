 
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
from Non_linear_lod.AdapComfunCacanove.ErrorIndicComp import big_indic
import ray

 
ray.init(num_cpus=48, include_dashboard=False)
#Fine_data
#____________________________________________________________________________________
Nh = 128                 # Number of divisions (mesh size is 1/N)
h = 1.0 / Nh                # Element size
FinSize = (Nh + 1) ** 2     # Total number of nodes in the mesh
num_Fin_elements = Nh ** 2  # Total number of elements in the mesh
u0=np.zeros(FinSize)        #  initial linearization point
#_________________________________________________________________________________
maxiter = 20
tol = 1e-12
#_____________________________________________________________________________________________________________
alpha1 = 1
beta=50
epslevel = 6
#______________________________________________________________________________________________________________
# Define the filename where the data will be saved
filename = 'saved_w.pkl'
#Since we have random data, one has to keep them in a file so we have consistent  results. 
# Check if the file already exists
if not os.path.exists(filename):
    # Generate the array
    w = np.round(np.random.rand(2**(2*epslevel) + 2**epslevel+ 1))
    # Save the array to a file
    with open(filename, 'wb') as f:
        pickle.dump(w, f)
    print("Array generated and saved.")
else:
    # Load the array from the file
    with open(filename, 'rb') as f:
        w = pickle.load(f)

epsilon1 = 2**(-epslevel)

def c(x):
    floor0 = np.floor((2**epslevel * x[:, 0])).astype(int)
    floor1 = np.floor((2**epslevel * x[:, 1])).astype(int)

    full_term = alpha1 * np.ones((x.shape[0])) + 5*w[floor0 + 2**epslevel * floor1]+(beta - alpha1)*((x[:, 0] > 0.4) & (x[:, 0] < 0.8) & (np.abs(x[:, 0]**2 - x[:, 1]) < 2 * epsilon1))
    result = full_term.astype(float)
    return result
#___________________________________________________________________________________________________________

#Definition of some nonlinear terms, 
#___________________________________________________________________________________________________________
alpha=0.005

# Exponential model ,Haverkamp model, VAn Genuchten

def O(s):
    return np.exp(2*s)
    #return 1/(1+(np.abs(s)*1)**1)
    #return ((1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2/(1+(alpha*np.abs(s))**2))

def Anonlin(x,s):
    return c(x)*O(s)
#the right-hand side used in most experiments
def right_hand_side1(x, y):
    if y<=0.35:
        return 2**4
    else:
        return 0.5
#________________________________________________________________________________________________________________
#different right hand side used in the third subsection of numerical experiment sections

#def right_hand_side(x, y):
#    if (y<=0.1):
#        return 0.1
#    else:
#        return 1
#def right_hand_side1(x, y):
#    if y<=0.15:
#        return 2**1
#    else:
#        return 0.1

#def right_hand_side2(x, y):
#    if y<=0.15:
#        return 2**2
#    else:
#        return 0.1


#def right_hand_side4(x, y):
#    if y<=0.15:
#        return 2**6
#    else:
#        return 0.1

#def right_hand_side5(x, y):
    
#    if y<=0.15:
#        return 2**8
#    else:
#        return 0.1

#def right_hand_side6(x, y):
    
#    if y<=0.15:
#        return 2**10
#    else:
#        return 0.1

#def right_hand_side7(x, y):
#    if y<=0.15:
#        return 2**12
#    else:
#        return 0.1

#def right_hand_side8(x,y):
#    if y<=0.15:
#        return 2**14
#    else:
#        return 0.1

  
#________________________________________________________________________________________________________________
#We compute all the corrections 

def global_correction(size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,stf,Mas):
# The needed data for computation of global corrector matrix
    CorSize=(N+1)**2
    B=np.eye((CorSize))
    msk=np.ones(CorSize,dtype='float')
    msk[C_bnd]=0
    B=diags(msk, format='csr')
    Ch =csc_matrix(pro @ Mas)
    # Submit tasks for each coarse element in parallel
    Batch=64
    results=[]
    for start in range (0, C_ele.shape[0], Batch):
        futures = [Quara.remote(l,size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro,stf,B,Ch) for l in range(start,min(start+Batch,C_ele.shape[0]))]
        batch_result=ray.get(futures)
        results.extend(batch_result)
    return results

#_______________________________________________________________________________________________________________
#Error indicator for a given element

def new_updated(size, uN1,uN2,Q_updated,pro,Stiff_new,C_ele,C_Nod, F_Nod, Tol,Anon):
    # Submit tasks for each coarse element in parallel
    futures = [big_indic.remote(l,size, uN1,uN2,Q_updated,lil_matrix(pro),lil_matrix(Stiff_new),C_ele,C_Nod, F_Nod, Tol,Anon) for l in range(C_ele.shape[0])]
    results=ray.get(futures)
    return results
def global_correction_update(size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,stf,Mas,M):
# The needed data for computation of global corrector matrix
    CorSize=(N+1)**2
    
    msk=np.ones(CorSize,dtype='float')
    msk[C_bnd]=0
    B=diags(msk, format='csr')
    Ch =csc_matrix(pro @ Mas)
     
    futures = [Quara.remote(l,size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro,stf,B,Ch) for l in M]
    # Collect results from each future and accumulate into the global matrix Qh
    resu=ray.get(futures)
    return resu
 
#_____________________________________________________________________________________________________

#Nonlinear solver of the PDE based on the new constructed space
def Non_Linear(size, N,maxit,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,Tol,right_hand_side):
        res = np.inf
        it = 0
        CorSize=(N+1)**2
        B=np.eye((CorSize))
        for i in C_bnd:
            B[i,i]=0
        Stiff15 , fh, Mass15 = assemble_system(F_Nod, F_ele, Nh,u_str,right_hand_side,Anon)

        free_index=np.array([i for i in range(B.shape[0]) if i not in  C_bnd ])
        Ch =csc_matrix(pro @ Mass15)
        Stiff125 , _, _ = assemble_system(F_Nod, F_ele, Nh,u0,right_hand_side,Anon)
        Modi_basis=[]
        while res > tol and it < maxit:
            print('computing correctors', end='', flush=True)   # The given functin in the lienarize step is the interpolation values.   ///
    
            if it==0:
                M=[]
                C=csr_matrix((CorSize,FinSize))
                Ce=global_correction(size, N,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,Stiff15,Mass15)
                for result in Ce:
                    C+=result
                clean_M=[]
                B_pr_C=B@(pro+C)
                pr_C= pro+C
                fH=B_pr_C@fh
                AH=B_pr_C @ Stiff15 @pr_C.T @ B
                 
                AH_reduced=AH[np.ix_(free_index,free_index)]
                fH_reduced=fH[free_index]
                u_free=np.linalg.solve(AH_reduced,fH_reduced)
                u_full=np.zeros_like(fH)
                u_full[free_index]=u_free
                ulod=pr_C.T @ u_full
                for i in F_bnd:
                    ulod[i]=0
                #print("number of basis that need modifying is:", len(clean_M))
            else:
                if it==1:
                    uN2=u_str 
                    uN1=ulodNew
                else:
                    uN2=uN1.copy()
                    uN1=ulodNew
                Ce_updated=Ce.copy()
                Q_updated = C.copy()
                if it==1:
                    previous_length=500000
                    current_length=5500000
                    pass
                else:
                    previous_length=current_length
                    current_length=length
                    
                if previous_length==0 and current_length==0:
                    pass
                else: 
                    M=new_updated(size, uN1,uN2,Q_updated,pro,Stiff125,C_ele,C_Nod, F_Nod, Tol,Anonlin)
        
                    clean_M = [x for x in M if x is not None]
             
                    if len(clean_M) !=0:
               
                        Ce_upd = global_correction_update(size, N, ulodNew, Anon, C_ele, C_Nod, F_ele, F_Nod, C_bnd, F_bnd, pro,Stiff_new, Mass15, clean_M)
                
                        for idx, new_val  in zip(clean_M,Ce_upd): 
                 
                            Q_updated+=new_val-Ce_updated[idx]
                            Ce_updated[idx]=new_val
                  
                Ce=Ce_updated.copy()
                C=Q_updated.copy()
                B_pr_C=B@(pro+C)
                pr_C= pro+C
                fH=B_pr_C @fh
                AH_reduced=KmsFree_new
                fH_reduced=fH[free_index]
                u_free=np.linalg.solve(AH_reduced,fH_reduced)
                u_full=np.zeros_like(fH)
                u_full[free_index]=u_free
                ulod=pr_C.T @ u_full
                for i in F_bnd:
                    ulod[i]=0
            #compute/update residual of nonlinear equation for stopping criterion
            Stiff_new,fh,Mass15=assemble_system(F_Nod, F_ele, Nh,ulod,right_hand_side,Anonlin)
            KFull_new = B_pr_C @ Stiff_new @pr_C.T @ B
            KmsFree_new = KFull_new[np.ix_(free_index,free_index)]
            res=np.linalg.norm(KmsFree_new@u_free-fH_reduced)
            ulodNew=ulod
            print('residual in {}th iteration is {}'.format(it, res), end='\n', flush=True)
            print( 'number of corrected basis=', len(clean_M))
            #print('residual in {}th iteration is {}'.format(it, res), end='\n', flush=True)
            Modi_basis.append(len(clean_M))
            it+=1
            length=len(clean_M)
        print('number of iterations=', it)
        return u_full,ulod,Modi_basis
#________________________________________________________________________________________________________________
def g(x, y):
    return 0.5*x * (1 - x) * y * (1 - y)*np.exp(5*(x+y))
    #return 10*x*y*(1-x)*(1-y)

CNodes1,CElements1, CBoundary1, FNodes1, FElements1, FBoundary1=Grid(2,Nh)
#___________________________________________________________________________________________________
#M12=new_updated(2, u0,uFullref,Q_updated,pro,Stiff_new,C_ele,C_Nod, F_Nod, Tol):
#___________________________________________________________________________________________________
CNodes12,CElements12, CBoundary12, FNodes12,  FElements12, FBoundary12=Grid(16,Nh)
ucoarse=solveFEM(np.zeros((16+1)**2),CNodes12, CElements12, CBoundary12,16,right_hand_side1,Anonlin,maxiter,tol)

Ph12= prolongation_matrix(CElements12, CNodes12, FNodes12)
u_st1=ucoarse@Ph12
#u_st1=np.array([g(FNodes1[i,0],FNodes1[i,1]) for i in range((Nh+1)**2)])
#_____________________________________________________________________________________________
#Convergence studies based on different   linearization points
@ray.remote(num_cpus=2)
def conver_history(kq,right_hand_side):
    H1=[]
  
    No_basis=[]
    CNodes1,CElements1, CBoundary1, FNodes1, FElements1, FBoundary1=Grid(2,Nh)
    uFullref=solveFEM(u0,FNodes1, FElements1, FBoundary1,Nh,right_hand_side,Anonlin,maxiter,tol)
    #___________________________________________________________________________________________________

    # p represent different sizes of coarse meshes
    for p in [2,4,8,16]:
        NH = p
        if p==2:
            Stiff12 , _, Mass1 = assemble_system(FNodes1, FElements1, Nh,u0,right_hand_side)
            ## Initialize Ph as a sparse matrix
            Ph= prolongation_matrix(CElements1, CNodes1, FNodes1)
            #ucoarse=solveFEM(np.zeros((p+1)**2),CNodes1, CElements1, CBoundary1,p,right_hand_side,Anonlin,maxiter,tol)
            #u_st1=ucoarse@Ph

            uH, ulod,Modi_basis= Non_Linear(kq, NH,20,u_st1,Anonlin, CElements1,CNodes1, FElements1, FNodes1 , CBoundary1, FBoundary1,Ph,0.1,right_hand_side)

            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            #Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
            #print('relative L2 error {}'.format(Mac_error))
            H1.append(H1semi)
       
            No_basis.append(Modi_basis)
        else:

            CNodes,CElements, CBoundary, FNodes,  FElements, FBoundary=Grid(NH,Nh)
            #ucoarse=solveFEM(np.zeros((NH+1)**2),CNodes, CElements, CBoundary,NH,right_hand_side,Anonlin,maxiter,tol)

            Ph= prolongation_matrix(CElements, CNodes, FNodes)
            #u_st1=ucoarse@Ph
            uH, ulod,Modi_basis= Non_Linear(kq, NH,20,u_st1,Anonlin, CElements,CNodes, FElements, FNodes , CBoundary,FBoundary, Ph,0.1, right_hand_side)
            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            #Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
        
            H1.append(H1semi)
          
            No_basis.append(Modi_basis)
        #Hf2.append(Mac_error)
    return H1, No_basis
#_______________________________________________________________________________________________________________

def global_exper(J,right_hand_side):
# The needed data for computation of global corrector matrix
    H_Errors=[]
  
    basis_1=[]
    # Submit the locaization parameters 
    futures1 = [conver_history.remote(kg,right_hand_side) for kg in [J]]
        # Collect results from each future 
    results1 =ray.get(futures1)

    for reslut,  basis in results1:
        H_Errors.append(reslut)
     
        basis_1.append(basis)
    return H_Errors,  basis_1
    #return results1

#Convergence studies for different sizes of the patch 
 
#for f in [right_hand_side1,right_hand_side2,right_hand_side3,right_hand_side4,right_hand_side5,right_hand_side6, right_hand_side7,right_hand_side8]:
for i in [3]:
    rel_upscaled_error , basis=global_exper(i,right_hand_side1)
    #rel_upscaled_error=global_exper(i,right_hand_side1)
    print(rel_upscaled_error)
 
    print(basis)
#________________________________________________________________________________________________________________





[[np.float64(0.6685535596226791), np.float64(0.21157390154499153), np.float64(0.09357546805886102), np.float64(0.03689709544905997)]]
[[[0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 16, 16, 12, 0, 0, 0, 0, 0, 0, 0], [0, 64, 63, 55, 0, 0, 0, 0, 0, 0], [0, 256, 200, 109, 0, 0, 0, 0, 0, 0]]]
