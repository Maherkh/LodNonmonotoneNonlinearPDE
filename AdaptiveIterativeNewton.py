 #Clean AdaptiveIterative Newton
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import splu,eigsh, eigs
from scipy.linalg import eig

import pickle
import os
import sys
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights
import scipy.sparse as sparse
import Non_linear_lod
from Non_linear_lod.FEM_solver.FEM_Matrices import assemble_system, Conv_grad
from Non_linear_lod.Prolongtion_Assembly import prolongation_matrix
from Non_linear_lod.Grid_Construction import Grid
from Non_linear_lod.FEM_solver.FEM_solver_Frechet import solve_Fem
from Non_linear_lod.FEM_solver.FEM_Solver_Cacanove import solveFEM
from Non_linear_lod.Correction_computation.Correction_Computation_Frechet import Quara
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights
from Non_linear_lod.Patch_info.Patch_Construction import find_patch
from Non_linear_lod.AdapCompfunNewton.Error_indicator_computations import big_indic
os.environ["RAY_DISABLE_EXPORT_AGENT"]="1"
import ray
ray.init(num_cpus=48, include_dashboard=False)
#____________________________________________________________
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
filename = 'saved_w.pkl '
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

#Definition of some nonlinear terms 
#___________________________________________________________________________________________________________

 
alpha=0.005
 
#O=lambda s:1/(1+(np.abs(s)*1)**1)
#O=lambda s:1/(1+(np.abs(s)*0.1)**0.5)


# Exponential model, Haverkamp model, VAn Genuchten
 
def O(s):
    return np.exp(2*s)
    #return 1/(1+(np.abs(s)*1)**1)
    #return ((1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2/(1+(alpha*np.abs(s))**2))

 
def Anonlin(x,s):
    return c(x)*O(s)

#def Anonlin_du(x,s):
#    return 2*c(x)*O(s)


def Anonlin_du(x,s):
    #K1=(1+(alpha*s)**2)**0.5
    #K2=(1+(alpha*s)**2)**2
    #dK1_ds=(1+(alpha*s)**2)**(3/2)*alpha**2*s
    #dK2_ds=4*(1+alpha*s)*alpha**2*s
    #derv=((2*K2*(K1-alpha*s)*(dK1_ds-alpha))-(K1-alpha*s)**2*dK2_ds)/(K2)**2
    #return c(x)*derv
    #return -1/(1+np.abs(s))
    return 2*c(x)*O(s)

#________________________________________________________________________________________________________________
#the right-hand side used in most experiments
def right_hand_side1(x, y):
    if (y<=0.35):
        return 2**4
    else:
        return 0.5
#________________________________________________________________________________________________________________

#We compute all the corrections 

def global_correction(size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod,C_bnd,F_bnd , pro,St,Ma,Co):
   #The needed data for computation of global corrector matrix
    CorSize=(N+1)**2
    FinSize=(Nh+1)**2
    msk=np.ones(CorSize, dtype='float')
    msk[C_bnd]=0
    B=diags(msk,format='csr')
    Ch =csc_matrix(pro @ Ma)
    Batch=40
    results=[]
    for start in range(0,C_ele.shape[0], Batch):
        futures = [Quara.remote(l,size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro, St,Ma,Co,B, Ch) for l in range(start, min(start+Batch,C_ele.shape[0]))]
        batch_result=ray.get(futures)
        results.extend(batch_result)
    return results

#___________________________________________________________________________________________________

# Gradient calculation of the discrete function 
# u_st


def new_updated(size, uN1,uN2,Q_updated,pro,Stiff_new,Mas, C_ele,C_Nod, F_Nod,F_ele, Tol):

    A_uN1=Anonlin(F_Nod,uN1)
    A_uN2=Anonlin(F_Nod,uN2)
    Adu_uN2=Anonlin_du(F_Nod,uN2)
    Adu_uN1=Anonlin_du(F_Nod,uN1)

    Batch=40
    results=[]
    for start in range(0,C_ele.shape[0], Batch):
        futures = [big_indic.remote(l,size, uN1,uN2,Q_updated,lil_matrix(pro),lil_matrix(Stiff_new),lil_matrix(Mas), C_ele,C_Nod, F_Nod,F_ele, Tol,A_uN1,A_uN2,Adu_uN1, Adu_uN2) for l in range(start, min(start+Batch,C_ele.shape[0]))]
        batch_result=ray.get(futures)
        results.extend(batch_result)

    return results

#this is to recompute the basis that requiring updates

def global_correction_update(size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod,C_bnd,F_bnd , pro,St,Ma,Co, M):
# The needed data for computation of global corrector matrix
    CorSize=(N+1)**2
    msk=np.ones(CorSize, dtype='float')
    msk[C_bnd]=0
    B=diags(msk,format='csr')

    Ch =pro @ Ma
    Ch=Ch.tocsr()

    
    Batch=40
    resu=[]
    for start in range(0,len(M), Batch):
        batch=M[start: start+Batch]
      
        futures = [Quara.remote(l,size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro, St,Ma,Co,B, Ch)  for l in batch]

        batch_result=ray.get(futures)
        resu.extend(batch_result)
   
    return resu

 

 
#_____________________________________________________________________________________________________

#The main code to run Adaptive iterative Algorithm in the context of Newton method


def Non_Linear(size,n ,N,maxit,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro, right_hand_side,Stiff125, Tol):
        
        Stiff15 , fh, Mass15 = assemble_system(F_Nod, F_ele, n,u_str,right_hand_side,Anon)
       
        res = np.inf
        it = 0
        CorSize=(N+1)**2
        msk=np.ones(CorSize, dtype='float')
        msk[C_bnd]=0
        B=diags(msk,format='csr')

        Convection , RHS_New =Conv_grad(n,F_ele, F_Nod,u_str , Anonlin_du)
        
        free_index=np.array([i for i in range(B.shape[0]) if i not in  C_bnd ])
        Modi_basis=[]
       

        while res > tol and it < maxit:
            print('computing correctors', end='', flush=True)   # The given functin in the lienarize step is the interpolation values.   ///
            #LOD solve
            if it==0:
                M=[]
                C=lil_matrix((CorSize,FinSize))
                Ce =global_correction(size, N ,u_str,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro, Stiff15,Mass15,Convection)
                for result in Ce:
                    C+=result
                C=C.tocsr()
                clean_M=[]
                B_p_C=B@(pro+C)
                p_C=(pro+C)
            
                fH=B_p_C @ (fh)
                AH=B_p_C @ (Stiff15+Convection)@p_C.T @ B
         
                AH_reduced=AH[np.ix_(free_index,free_index)]
                fH_reduced=fH[free_index]
              
                
                AH_res=B_p_C @ (Stiff15) @p_C.T @ B
                 
                uHH=spsolve((pro@Mass15@pro.T)[np.ix_(free_index,free_index)],(pro@Mass15@u_str)[free_index])  
                resd_red=-(AH_res[np.ix_(free_index,free_index)]@uHH-fH[free_index])
          
               
                #to solve the system one has to creat the mask for the non-boundary points
                u_free=spsolve(AH_reduced,resd_red)
                u_full=np.zeros_like(fH)
                u_full[free_index]=u_free
            
                delta=p_C.T @ u_full
                fH_reduced=fH[free_index]
                 
                usst=np.zeros_like(fH)
                usst[free_index]= uHH
                ulod=(pro+C).T@usst +delta
                res=np.linalg.norm(delta)
                ulod[F_bnd]=0

            else:
                if it==1:
                    uN2=u_str 
                    uN1=ulodNew
                else:
                    uN2=uN1.copy()
                    uN1=ulodNew
                Ce_updated=list(Ce)
                Q_updated = C
                if it==1:
                    previous_length=500000
                    current_length=5500000
                    pass
                else:
                    previous_length=current_length
                    current_length=length
                    
                if not ( previous_length==0 and current_length==0):
                    M=new_updated(size, uN1,uN2,Q_updated,pro,Stiff125,Mass15, C_ele,C_Nod, F_Nod,F_ele, Tol)
                    clean_M = [x for x in M if x is not None]
                
                    if clean_M:
                        Ce_upd=global_correction_update(size, N ,ulodNew,Anon,A_du, C_ele,C_Nod, F_ele, F_Nod,C_bnd,F_bnd , pro,Stiff_new,Mass15,Conv_new, clean_M)
                        for idx, new_val  in zip(clean_M,Ce_upd):
                            old_val=Ce_updated[idx]
                            delta=new_val-old_val
                            Q_updated+=delta
                            Ce_updated[idx]=new_val
                Ce=Ce_updated
                C=Q_updated.copy()
             
                B_pr_C=B@(pro+C)
                pr_C= pro+C
                fH=B_pr_C @fh
                AH_reduced=KmsFree_new
                fH_reduced=fH[free_index]
      
                u_full=np.zeros_like(fH)
             
                
                
                KFull_res = B_p_C@(Stiff_new)@p_C.T @ B
                KmsFree_new1 = KFull_res[np.ix_(free_index,free_index)] 
                
                uHH=spsolve((pro@Mass15@pro.T)[np.ix_(free_index,free_index)],(pro@Mass15@ulodNew)[free_index])  

                resright=-(KmsFree_new1@uHH-fH_reduced)
        
                AH=B_p_C @ (Stiff_new+Conv_new) @p_C.T @ B
       
                AH_reduced=AH[np.ix_(free_index,free_index)]
                fH_reduced=fH[free_index]
                
         
                u_free=spsolve(AH_reduced,resright)
                u_full=np.zeros_like(fH)
                u_full[free_index]=u_free
            
                delta=p_C.T @ u_full
                res=np.linalg.norm(delta)
                ulod=ulodNew+delta
                ulod[F_bnd]=0

            #compute/update residual of nonlinear equation for stopping criterion
            Stiff_new, _, _ =assemble_system(F_Nod, F_ele, Nh,ulod,right_hand_side ,Anonlin)

            Conv_new, _=Conv_grad(n,F_ele, F_Nod,ulod , Anonlin_du)
            KFull_new = B_p_C@(Stiff_new+Conv_new )@p_C.T @ B
            KmsFree_new = KFull_new[np.ix_(free_index,free_index)]
            #res=np.linalg.norm(KmsFree_new@u_free-fH_reduced)
            ulodNew=ulod
            print('residual in {}th iteration is {}'.format(it, res), end='\n', flush=True)
            print( 'number of corrected basis=', len(clean_M))
            #print('residual in {}th iteration is {}'.format(it, delta), end='\n', flush=True)
        

            Modi_basis.append(len(clean_M))
            it+=1
            length=len(clean_M)
        print('number of iterations=', it)

        return u_full,  ulod,Modi_basis



#________________________________________________________________________________________________________________
def g(x, y):
    return 0.5*x * (1 - x) * y * (1 - y)*np.exp(5*(x+y))
    #return 10*x*y*(1-x)*(1-y)

CNodes1,CElements1, CBoundary1, FNodes1, FElements1, FBoundary1=Grid(2,Nh)
 
#___________________________________________________________________________________________________



#Convergence studies based on different   linearization points
@ray.remote(num_cpus=2)
def conver_history(kq,right_hand_side):
    H1=[]
    #L2=[]
    No_basis=[]
    #CNodes1,CElements1, CBoundary1, FNodes1, FElements1, FBoundary1=Grid(2,Nh)
    #uFullref=solve_Fem(Nh,u0,FNodes1, FElements1, FBoundary1,right_hand_side,Anonlin,Anonlin_du,maxiter,tol)
    uFullref=solveFEM(u0,FNodes1, FElements1, FBoundary1,Nh,right_hand_side,Anonlin,maxiter,tol)
    Stiff12 , _, Mass1 = assemble_system(FNodes1, FElements1, Nh,u0,right_hand_side)
    Stiff125 , _, _ = assemble_system(FNodes1, FElements1, Nh,u0,right_hand_side,Anonlin)

    #___________________________________________________________________________________________________

    # p represent different sizes of coarse meshes
    for p in [2,4,8,16]:
        NH = p
        if p==2:
                       ## Initialize Ph as a sparse matrix
            Ph= prolongation_matrix(CElements1, CNodes1, FNodes1)
            #ucoarse=solveFEM(np.zeros((p+1)**2),CNodes1, CElements1, CBoundary1,p,right_hand_side,Anonlin,maxiter,tol)
            #u_st1=ucoarse@Ph
                       
            uH, ulod,Modi_basis= Non_Linear(kq,Nh,p,20,u0,Anonlin,Anonlin_du, CElements1,CNodes1, FElements1, FNodes1 , CBoundary1,FBoundary1, Ph, right_hand_side,Stiff125, 0.1)
            
            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            #Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
            #print('relative L2 error {}'.format(Mac_error))
            H1.append(float(H1semi))
            #L2.append(float(Mac_error))
            No_basis.append(Modi_basis)
        else:

            CNodes,CElements, CBoundary, _,  _, _=Grid(NH,Nh)
            #ucoarse=solveFEM(np.zeros((NH+1)**2),CNodes, CElements, CBoundary,NH,right_hand_side,Anonlin,maxiter,tol)

            Ph= prolongation_matrix(CElements, CNodes, FNodes1)
            #u_st1=ucoarse@Ph
            uH, ulod,Modi_basis= Non_Linear(kq,Nh,p,20,u0,Anonlin,Anonlin_du, CElements,CNodes, FElements1, FNodes1 , CBoundary,FBoundary1, Ph, right_hand_side,Stiff125, 0.1)
            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            #Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
            #print('relative L2 error {}'.format(Mac_error))
            H1.append(float(H1semi))
            #L2.append(float(Mac_error))
            No_basis.append(Modi_basis)
        #Hf2.append(Mac_error)
    return H1,No_basis
#________________________________________________________________________________________________________________


def global_exper(J,right_hand_side):
# The needed data for computation of global corrector matrix
    H_Errors=[]
    #L2_Errors=[]
    basis_1=[]
    # Submit the locaization parameters 
    futures = [conver_history.remote(kg,right_hand_side) for kg in [J]]
        # Collect results from each future 
    results =ray.get(futures)
    for reslut, basis in results:
        H_Errors.append(reslut)
        #L2_Errors.append(err)
        basis_1.append(basis)
    return H_Errors, basis_1

#Convergence studies for different sizes of the patch 

for i in [3]:
    rel_upscaled_error, basis=global_exper(i,right_hand_side1)
    print(rel_upscaled_error)
   # print(rel_macro_error)
    print(basis)
    
