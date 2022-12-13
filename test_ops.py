#!/usr/bin/env python3
# Andrew Lytle
# Dec 2022

#import numpy as np
import pickle
import time
#from statsmodels.stats.weightstats import DescrStatsW
from math import log
import logging
import cProfile
import pstats

import scipy.sparse as sp

from decompose_pauli import to_pauli_vec
import numpy as np
#from e1plus import get_eigenvalues
from qiskit import Aer
import matplotlib.pyplot as plt
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA,COBYLA
from qiskit.circuit.library import TwoLocal,EfficientSU2
from qiskit.opflow.converters import AbelianGrouper #, NewAbelianGrouper


from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import X, Y, Z, I, CX, T, H, S, PrimitiveOp
from qiskit.opflow import (I, X, Y, Z, H, CX, Zero, ListOp, PauliExpectation, 
                           PauliTrotterEvolution, CircuitSampler, MatrixEvolution, 
                           Suzuki)
from qiskit.circuit import Parameter


from rmatrix import random_H
from timing import profile
from VQE_driver import array_to_Op, array_to_SummedOp
    
def main():
    m = 5

    N = 2**m
    Hmat = random_H(N)
    print(Hmat)

    evals,evec = np.linalg.eigh(Hmat)
    ref_value=evals[0]
    print("Exact value: ",evals[0])

     
    H = array_to_SummedOp(Hmat, m)
    print(H)
    sf = StateFn(H)
    print('Summed op eval:')
    print(sf.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
    sf = sf.adjoint()
    with profile():
        for _ in range(100):
            sf.eval(Zero^Zero^Zero^Zero^Zero)
     
   
    '''
    H = array_to_Op(Hmat) 
    grouper = AbelianGrouper()
    H = grouper.convert(H)
    sf2 = StateFn(H)
    print(sf2.adjoint())
    print('abelian eval:')
    print(sf2.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
    sf2 = sf2.adjoint()
    
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        sf2.eval(Zero^Zero^Zero^Zero^Zero)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats()
    '''
    
if __name__ == "__main__":
    main()
