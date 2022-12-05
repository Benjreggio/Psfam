#!/usr/bin/env python3
# Andrew Lytle
# Nov 2022

import numpy as np
from numpy.linalg import eig
from numpy.random import normal
from scipy.stats import unitary_group

def random_evals(N):
    "List of Gaussian distributed numbers [(0,1),...]."
    vals = []
    for i in range(N):
        vals.append(normal())
    return vals

def dot_all(Ms):
    "Dot product of [M1, M2, ...]"
    res = np.identity(Ms[0].shape[0])
    for M in Ms[::-1]:
        res = np.dot(M, res)
    return res

def hc(M):
    "Hermitian conjugate of M."
    return M.conj().T

def random_H(N):
    "Random NxN Hermitian matrix."
    evs = random_evals(N)
    D = np.diag(evs)
    U = unitary_group.rvs(N)
    H = dot_all([U, D, hc(U)])
    return H

def test():
    m=3
    N=pow(2,m)

    print(m)
    print(N)

    evs = random_evals(N)
    print(evs)

    D = np.diag(evs)
    print(D)

    U = unitary_group.rvs(N)

    H = dot_all([U, D, hc(U)])
    #print(H)

    vals, vecs = eig(H)
    print(vals)
    print(evs)

if __name__ == '__main__':
    test()
    
