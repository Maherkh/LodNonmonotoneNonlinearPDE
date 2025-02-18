#FINITE ELEMTN METHOD
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
# Here we creat the geometries of th problem, Fine and coarse meshes:
#from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import pickle
import os
import multiprocessing
import scipy.sparse as sparse
import ray

ray.init()
#____________________________________________________________
# #Gauss quadrature points
sq = np.sqrt(1/3)
gauss_points=np.array([(-sq,-sq),(sq,-sq),(-sq,sq),(sq,sq)])
weights=[1,1,1,1]

#Fine_data
#____________________________________________________________________________________
Nh = 128                    # Number of divisions (mesh size is 1/N)
h = 1.0 / Nh                # Element size
FinSize = (Nh + 1) ** 2     # Total number of nodes in the mesh
num_Fin_elements = Nh ** 2  # Total number of elements in the mesh
#_________________________________________________________________________________
 

#________________________________________________________________________________________________________________
 

#________________________________________________________________________________________________________________

maxiter = 10
tol = 1e-12


alpha1 = 1
beta=50
epslevel = 6
#______________________________________________________________________________________________________________
#Definition_of_connectivity_field_functions

#________________________________________________________________________________________________________________
# Define the filename where the data will be saved
filename = 'saved_w.pkl128'
#Since we have random data one has to keep them in a file so we have consistent compariosn and consistitent results. 
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

def f(x, y):
    if (y<=0.1):
        return 0.1
    else:
        return 1