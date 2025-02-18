import numpy as np
def Grid(coarse, fine):
    # Create mesh for coarse grid
    x_coarse = np.linspace(0, 1, coarse + 1)
    y_coarse = np.linspace(0, 1, coarse + 1)
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
    # Create mesh for fine grid
    x_fine = np.linspace(0, 1, fine + 1)
    y_fine = np.linspace(0, 1, fine + 1)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    #Enumerations of the nodes:
    def enumerate_points(X, Y):
        points = np.vstack([X.ravel(), Y.ravel()]).T
        point_indices = np.arange(points.shape[0])
        return points, point_indices

    #Enumerations of the element (Conectivity matrix):
    def enumerate_elements(X, Y):
        elements = []
        rows, cols = X.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                elements.append([
                    i * cols + j,
                    i * cols + (j + 1),
                    (i + 1) * cols + j,
                    (i + 1) * cols + (j+1)
                ])
        return np.array(elements)
    #Enumerations of border points:
    def enumerate_border_points(X, Y):
        rows, cols = X.shape
        border_points = set()

        # Top and bottom rows:
        for j in range(cols):
            border_points.add(j)  
            border_points.add((rows - 1) * cols + j)
        # Left and right columns:
        for i in range(rows):
            border_points.add(i * cols)  
            border_points.add(i * cols + (cols - 1)) 

        return np.array(list(border_points))
# initite the data of the CoarseMesh
    CoarseNodes, CoarseNodeIndex = enumerate_points(X_coarse, Y_coarse)
    CoarseElements = enumerate_elements(X_coarse, Y_coarse)
    CoarseBoundary = enumerate_border_points(X_coarse, Y_coarse)

    FineNodes, FineNodeIndex = enumerate_points(X_fine, Y_fine)
    FineElements = enumerate_elements(X_fine, Y_fine)
    FineBoundary = enumerate_border_points(X_fine, Y_fine)

    return CoarseNodes, CoarseElements, CoarseBoundary, FineNodes, FineElements, FineBoundary

