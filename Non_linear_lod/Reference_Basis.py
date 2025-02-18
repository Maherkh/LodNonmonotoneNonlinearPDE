import numpy as np 



sq = np.sqrt(1/3)
gauss_points=np.array([(-sq,-sq),(sq,-sq),(-sq,sq),(sq,sq)])
weights=[1,1,1,1]


def ref_basis(x, y):
    Phi = np.array([
        0.25*(1 - x) * (1 - y),
        0.25*(1+x)* (1 - y),
        0.25*(1 - x) * (1+y),
        0.25*(1+x) * (1+y)
    ])
    gradPhi = np.array([
        [-0.25*(1 - y), -0.25*(1 - x)],
        [0.25*(1 - y), -0.25*(1+x)],
        [-0.25*(1+y), 0.25*(1 - x)],
        [0.25*(1+y), 0.25*(1+x)]
    ])
    return Phi, gradPhi


