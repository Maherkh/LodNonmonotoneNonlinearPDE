import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from Non_linear_lod.Patch_info.Patch_Construction import find_patch


def Patch_matrices(l,m, connectivity1, C_Nod, F_Nod, C_bnd,F_bnd): # l is the index of the element,  m is the size of the patch

    CorSize=len(C_Nod[:,0])
    FinSize=len(F_Nod[:,0])
    Patch= find_patch(connectivity1,l,m)
    # Local Patch restrections Coarse:
    CorPatch_nodes=list(set((connectivity1[list(Patch)].flatten())))
    CorPatch_nodes.sort()
    NLH=[item for item in CorPatch_nodes if item not in C_bnd]
    RHl=np.zeros((len(NLH),CorSize))
    for i in range(len(NLH)):
        RHl[i,NLH[i]]=1

    finePatchNodes=set()
    coordCorNodes = np.array(C_Nod)
    for idx, Node in enumerate(F_Nod):
        if (coordCorNodes[min(CorPatch_nodes)][0] < Node[0] <coordCorNodes[max(CorPatch_nodes)][0] and coordCorNodes[min(CorPatch_nodes)][1] <Node[1] <coordCorNodes[max(CorPatch_nodes)][1]):
            finePatchNodes.add(idx)
    finePatchNodes1=list(finePatchNodes)
    finePatchNodes1.sort()
    NLh=[item for item in finePatchNodes1 if item not in F_bnd]
    # 
    Rhl=np.zeros((len(NLh),FinSize))
    for i in range(len(NLh)):
        Rhl[i,NLh[i]]=1
    return csc_matrix(RHl),  csc_matrix(Rhl),  len(NLh)
