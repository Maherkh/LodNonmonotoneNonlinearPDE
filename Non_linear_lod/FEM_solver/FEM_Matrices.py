import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from Non_linear_lod.Reference_Basis import ref_basis,gauss_points, weights 



def assemble_system(nodes, elements, N, u_star,right_hand,Anon=None):
    Siz = (N+1)**2
    A_matrix = lil_matrix((Siz, Siz))  # Global Stiffness matrix
    Mass = lil_matrix((Siz, Siz))        # Global Mass matrix
    b_vector= np.zeros(Siz)
    if Anon is not None:
        for element in elements:
            Fe = np.zeros(4)       # Local load vector
            ele_coords = nodes[element, :]
            u_star_elem=u_star[element]
            m_point=np.mean(ele_coords, axis=0)
            m_u=np.mean(u_star_elem, axis=0)
            A_star=Anon(np.array([(m_point[0],m_point[1])]),np.array([m_u]))
            for gp, w1 in zip(gauss_points, weights):
                x, y = gp
                Phi_val, grad_val = ref_basis(x, y)
                J = np.dot(grad_val.T, ele_coords)
                detJ = np.linalg.det(J)
                invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
                
                grad_N = np.dot(invJ, grad_val.T) # Gradient of basis in real coordinates
                x_gp = np.dot(Phi_val, ele_coords[:, 0])
                y_gp = np.dot(Phi_val, ele_coords[:, 1])
                for i in range(4):
                    for j in range(4):
                       
                        A_matrix[element[i], element[j]] += A_star[0]*np.dot(grad_N[:, i], grad_N[:, j]) * detJ * w1
                        Mass[element[i], element[j]] += Phi_val[i] * Phi_val[j] * detJ * w1
                f_val = right_hand(x_gp, y_gp)
                for r in range(4):
                    Fe[r] +=  f_val *Phi_val[r] * detJ * w1
            for i in range(4):
                b_vector[element[i]] += Fe[i]
    else:
        for element in elements:
            
            ele_coords = nodes[element, :]
            for gp, w1 in zip(gauss_points, weights):
                x, y = gp
                Phi_val, grad_val = ref_basis(x, y)

                J = np.dot(grad_val.T, ele_coords)
                detJ = np.linalg.det(J)
                invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

                grad_N = np.dot(invJ, grad_val.T)

                for i in range(4):
                    for j in range(4):
                        A_matrix[element[i],element[j]] += np.dot(grad_N[:, i], grad_N[:, j]) * detJ * w1
                        Mass[element[i],element[j]] += Phi_val[i] * Phi_val[j] * detJ * w1
                        
    return csc_matrix(A_matrix), b_vector, csc_matrix(Mass)


# we create the convection term
def Conv_grad(N,elements, nodes, u_str, A_du):
    Siz=(N+1)**2
    Conv = lil_matrix((Siz, Siz))  
    b_str=np.zeros(Siz)
    for element in elements:
        ele_coords = nodes[element, :]
        u_star_elem=u_str[element]
        m_point=np.mean(ele_coords, axis=0)
        m_u=np.mean(u_star_elem, axis=0)
        A_u=A_du(np.array([(m_point[0],m_point[1])]),np.array([m_u]))
        for gp, w1 in zip(gauss_points, weights):
            xi, yi = gp
            Phi_val, grad_val_ref = ref_basis(xi, yi)

            J = np.dot(grad_val_ref.T, ele_coords)  
            detJ = np.linalg.det(J)
            invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
            grad_val_phys = np.dot(invJ, grad_val_ref.T).T  
            u_star_gp = np.dot(Phi_val, u_str[element])  # Interpolated u_star at Gauss point
            grad_u_star_ref = np.dot(grad_val_ref.T, u_str[element])  
            grad_u_star_phys = np.dot(invJ.T, grad_u_star_ref)  
            scalar_term =   (u_star_gp * A_u)[0] * grad_u_star_phys
            for i in range(4):  
                for j in range(4):  
                    grad_Phi_j_phys = grad_val_phys[j, :]  # Grad(Phi_j) in physical space
                    convective_term =  A_u[0] * grad_u_star_phys  
                    Conv[element[i],element[j]] += Phi_val[i] * np.dot(convective_term, grad_Phi_j_phys) * detJ * w1
        for i in range(4):
            b_str[element[i]] += np.dot(scalar_term, grad_val_phys[i, :]) * detJ * w1

    return csc_matrix(Conv), b_str