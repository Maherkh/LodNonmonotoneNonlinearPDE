import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

def prolongation_matrix(C_ele, C_Nod, F_Nod):
    NH=int(np.sqrt(C_ele.shape[0]))
    H=1/NH
    CorSize=(NH+1)**2
    FinSize=F_Nod.shape[0]
    Pro= lil_matrix((CorSize, FinSize))
    for i in range(len(C_ele)):
        # x &y coordinates for coarse nodes
        x1 = C_Nod[C_ele[i][0]][0]
        x2 = C_Nod[C_ele[i][1]][0]

        y1 = C_Nod[C_ele[i][0]][1]
        y4 = C_Nod[C_ele[i][3]][1]
        LO = C_ele[i, :]
        # Vectorized basis functions over all FineNodes
        def phi1(X):
            cond = (x1 <= X[0]) & (X[0] < x2) & (y1 <= X[1]) & (X[1] < y4)
            return np.where(cond, (x2-X[0])*(y4-X[1])/(H**2), 0)

        def phi2(X):
            cond = (x1 < X[0]) & (X[0] <= x2) & (y1 <= X[1]) & (X[1] < y4)
            return np.where(cond, (X[0]-x1)*(y4-X[1])/(H**2), 0)

        def phi4(X):
            cond = (x1 < X[0]) & (X[0] <= x2) & (y1 < X[1]) & (X[1] <= y4)
            return np.where(cond, (X[0]-x1)*(X[1]-y1)/(H**2), 0)

        def phi3(X):
            cond = (x1 <= X[0]) & (X[0] < x2) & (y1 < X[1]) & (X[1] <= y4)
            return np.where(cond,(x2-X[0])*(X[1]-y1)/(H**2), 0)
        
        functions = [phi1, phi2, phi3, phi4]   # List of function references  

        # Evaluate all phi functions on all fine nodes at once (vectorized)
        coarse_start = C_Nod[LO[0]]
        coarse_end = C_Nod[LO[3]]
        # Find indices in FineNodes that match the coarse start and end points
        start_idx = np.where(np.isclose(F_Nod[:, 0], coarse_start[0]) & np.isclose(F_Nod[:, 1], coarse_start[1]))[0][0]
        end_idx = np.where(np.isclose(F_Nod[:, 0], coarse_end[0]) & np.isclose(F_Nod[:, 1], coarse_end[1]))[0][0]
        # Loop through the limited range of FineNodes between start_idx and end_idx
        for j in range(4):
            for k in range(start_idx, end_idx+1):
                if Pro[LO[j], k]==0:
                    Pro[LO[j], k] = functions[j](F_Nod[k, :])
    return csc_matrix(Pro)