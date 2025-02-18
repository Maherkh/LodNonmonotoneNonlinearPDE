import numpy as np
def find_patch(connectivity, element_index, k):
        def get_neighbors(connectivity, element_set):
            neighbors = set()
            for idx, elem in enumerate(connectivity):
                if any(np.isin(elem, connectivity[list(element_set)].flatten())):
                    neighbors.add(idx)
            return neighbors
        element_set = {element_index}
        patch = element_set.copy()
        for i in range(k):
            new_neighbors = get_neighbors(connectivity, element_set)
            patch.update(new_neighbors)
            element_set = new_neighbors
        return patch