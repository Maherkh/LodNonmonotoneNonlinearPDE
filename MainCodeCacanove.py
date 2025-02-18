#FINITE ELEMTN METHOD
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

import pickle
import os
import sys

import scipy.sparse as sparse
import Non_linear_lod
from Non_linear_lod.FEM_solver.FEM_Matrices import assemble_system
from Non_linear_lod.Prolongtion_Assembly import prolongation_matrix
from Non_linear_lod.Grid_Construction import Grid
from Non_linear_lod.FEM_solver.FEM_Solver_Cacanove import solveFEM
from Non_linear_lod.Correction_computation.Correction_Computation_Cacanove import Quara
from Non_linear_lod.Reference_Basis import ref_basis, gauss_points, weights
import ray

ray.init()
#____________________________________________________________


#Fine_data
#____________________________________________________________________________________
Nh = 64                    # Number of divisions (mesh size is 1/N)
h = 1.0 / Nh                # Element size
FinSize = (Nh + 1) ** 2     # Total number of nodes in the mesh
num_Fin_elements = Nh ** 2  # Total number of elements in the mesh
u0=np.zeros(FinSize)        #  initial linearization point
#_________________________________________________________________________________
maxiter = 10
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

#VAn Genuchten
alpha=0.005

#O=lambda s:(1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2/(1+(alpha*np.abs(s))**2)
#Os=lambda s: (1+(alpha*np.abs(s))**2)*(2*(1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2)))))*(-0.5*alpha*np.abs(s)*(1+(alpha*np.abs(s))**2)**(-3/2)*2*alpha*np.abs(s)*np.sign(s)-(alpha*np.sign(s)*(1+alpha*np.abs(s))**2)**-0.5)-((1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2)*2*alpha*np.abs(s)*np.sign(s)/(1+(alpha*np.abs(s))**2)**2
#Os=lambda s: (1+(alpha*np.abs(s))**2)*(2*(1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2)))))*(-0.5*alpha*np.abs(s)*(1+(alpha*np.abs(s))**2)**(-3/2)*2*alpha*np.abs(s)*np.sign(s)-(alpha*np.sign(s)*(1+alpha*np.abs(s))**2)**-0.5)-((1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2)*2*alpha*np.abs(s)*np.sign(s)/(1+(alpha*np.abs(s))**2)**2


#Haverkamp model
#O=lambda s:1/(1+(np.abs(s)*1)**1)
#O=lambda s:1/(1+(np.abs(s)*0.1)**0.5)


# Exponential model
#def O(s):
#return (1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2/(1+(alpha*np.abs(s))**2)
#O=lambda s:np.exp(2*s)
#O=lambda s:np.exp(1*s/1)

def O(s):
    return np.exp(2*s)

#O=lambda s:((1-(alpha*np.abs(s)/(np.sqrt(1+(alpha*np.abs(s))**2))))**2/(1+(alpha*np.abs(s))**2))

#Anonlin = lambda x, s:  c(x)*O(s)
def Anonlin(x,s):
    return c(x)*O(s)

#________________________________________________________________________________________________________________

def right_hand_side(x, y):
    if (y<=0.1):
        return 0.1
    else:
        return 1
#________________________________________________________________________________________________________________
  
#________________________________________________________________________________________________________________
def global_correction(size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,stf,Mas):
# The needed data for computation of global corrector matrix
    CorSize=(N+1)**2
    Ql = csr_matrix((CorSize, FinSize))  # Initialize the correction matrix
    B=np.eye((CorSize))
    for i in C_bnd:
        B[i,i]=0
    B=csc_matrix(B)
    Ch =csc_matrix(pro @ Mas)
    # Submit tasks for each coarse element in parallel
    futures = [Quara.remote(l,size, N ,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd, F_bnd,pro,stf,B,Ch) for l in range(C_ele.shape[0])]

    # Collect results from each future and accumulate into the global matrix Qh
    results=ray.get(futures)
    for reslut in results:
        Ql+=reslut
    return csr_matrix(Ql)
#_______________________________________________________________________________________________________________

#Nonlinear solver of the PDE based on the new constructed space
def Non_Linear(size, N,maxit,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro):
        res = np.inf
        it = 0
        CorSize=(N+1)**2
        B=np.eye((CorSize))
        for i in C_bnd:
            B[i,i]=0
        Stiff15 , fh, Mass15 = assemble_system(F_Nod, F_ele, Nh,u_str,right_hand_side,Anon)
        C=global_correction(size, N,u_str,Anon, C_ele,C_Nod, F_ele, F_Nod , C_bnd,F_bnd, pro,Stiff15,Mass15)
        B_pr_C=B@(pro+C)
        pr_C= pro+C
        free_index=np.array([i for i in range(B.shape[0]) if i not in  C_bnd ])

        while res > tol and it < maxit:
            print('computing correctors', end='', flush=True)   # The given functin in the lienarize step is the interpolation values.   ///
            #LOD solve
            if it==0:
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
            else:
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
            Stiff_new,_,_=assemble_system(F_Nod, F_ele, Nh,ulod,right_hand_side,Anonlin)
            KFull_new = B_pr_C @ Stiff_new @pr_C.T @ B
            KmsFree_new = KFull_new[np.ix_(free_index,free_index)]
            res=np.linalg.norm(KmsFree_new@u_free-fH_reduced)

            print('residual in {}th iteration is {}'.format(it, res), end='\n', flush=True)
            it +=1
        return u_full,ulod
#________________________________________________________________________________________________________________
def g(x, y):
    return 0.5*x * (1 - x) * y * (1 - y)*np.exp(5*(x+y))
   #return 10*x*y*(1-x)*(1-y)

CNodes1,CElements1, CBoundary1, FNodes1, FElements1, FBoundary1=Grid(2,Nh)
uFullref=solveFEM(u0,FNodes1, FElements1, FBoundary1,Nh,right_hand_side,Anonlin,maxiter,tol)
 
#_____________________________________________________________________________________________
#Convergence studies based on different   linearization points
@ray.remote
def conver_history(kq):
    H1=[]
    L2=[]
     
    for p in [2,4,8,16,32]:
        NH = p
        if p==2:
             

            Stiff12 , _, Mass1 = assemble_system(FNodes1, FElements1, Nh,u0,right_hand_side)
            
            # Initialize Ph as a sparse matrix
            Ph= prolongation_matrix(CElements1, CNodes1, FNodes1)
            uH, ulod= Non_Linear(kq, NH,10,u0,Anonlin, CElements1,CNodes1, FElements1, FNodes1 , CBoundary1, FBoundary1,Ph)

            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
            print('relative L2 error {}'.format(Mac_error))
            H1.append(H1semi)
            L2.append(Mac_error)
        else:

            CNodes,CElements, CBoundary, FNodes,  FElements, FBoundary=Grid(NH,Nh)
            Ph= prolongation_matrix(CElements, CNodes, FNodes)
            uH, ulod= Non_Linear(kq, NH,10,ulod,Anonlin, CElements,CNodes, FElements, FNodes , CBoundary,FBoundary, Ph)
            H1semi = np.sqrt(np.dot((uFullref -ulod ),  Stiff12@(uFullref -ulod)))/np.sqrt(np.dot(uFullref,  Stiff12@uFullref))
            Mac_error = np.sqrt(np.dot(uFullref - Ph.T@uH, Mass1 * (uFullref -Ph.T@uH))) / np.sqrt(np.dot(uFullref, Mass1 * uFullref))
            print('relative H1semi error {}'.format(H1semi))
            print('relative L2 error {}'.format(Mac_error))
            H1.append(H1semi)
            L2.append(Mac_error)
        #Hf2.append(Mac_error)
    return H1, L2

#________________________________________________________________________________________________________________

def global_exper(J):
# The needed data for computation of global corrector matrix
    H_Errors=[]
    L2_Errors=[]
    # Submit tasks for each coarse element in parallel
    futures = [conver_history.remote(kg) for kg in [J]]
        # Collect results from each future and accumulate into the global matrix Qh
    results =ray.get(futures)
    for reslut,err in results:
        H_Errors.append(reslut)
        L2_Errors.append(err)
    return H_Errors,L2_Errors

#Convergence studies for different sizes of the patch 
for i in [1]:
    rel_upscaled_error,rel_macro_error=global_exper(i)
    print(rel_upscaled_error)
    print(rel_macro_error)

#________________________________________________________________________________________________________________

