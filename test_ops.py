#!/usr/bin/env python3
# Andrew Lytle
# Dec 2022

#import numpy as np
import pickle
import time
#from statsmodels.stats.weightstats import DescrStatsW
from math import log
import logging
log = logging.getLogger(__name__)
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
from qiskit.test.mock.backends import FakeCasablanca

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
    run_new = False
    run_abelian = True

    m = 5

    N = 2**m
    Hmat = random_H(N)
    log.info(Hmat)

    evals,evec = np.linalg.eigh(Hmat)
    ref_value=evals[0]
    log.info(f"Exact value: {evals[0]}")

    if run_new: 
        H = array_to_SummedOp(Hmat, m)
        log.info(H)
        sf = StateFn(H)
        log.info('Summed op eval:')
        log.info(sf.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
        sf = sf.adjoint()
        #with profile():
        #    for _ in range(100):
        #        sf.eval(Zero^Zero^Zero^Zero^Zero)
        
        #backend=Aer.get_backend('aer_simulator')
        backend = FakeCasablanca()
        qi = QuantumInstance(backend, shots=1024)
        expectation = PauliExpectation().convert(sf.compose(Zero^Zero^Zero^Zero^Zero))
        with profile():
            sampler = CircuitSampler(qi, attach_results=True).convert(expectation)
        
            print(f'sampler: {sampler}')
            print('RESULT:', sampler.eval())
        #with profile():
        #    for _ in range(100):
        #        print(sampler.eval())

    
    #CircuitSampler (CircuitStateFunc, Circuit Sampler)
    
    if run_abelian:
        H = array_to_Op(Hmat) 
        grouper = AbelianGrouper()
        H = grouper.convert(H)
        sf2 = StateFn(H)
        log.debug(sf2.adjoint())
        log.info('abelian eval:')
        log.info(sf2.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
        sf2 = sf2.adjoint()
        #with profile(): 
        #    for _ in range(100):
        #        sf2.eval(Zero^Zero^Zero^Zero^Zero)
        
        #backend=Aer.get_backend('aer_simulator')
        backend = FakeCasablanca()
        qi = QuantumInstance(backend, shots=1024)
        expectation = PauliExpectation().convert(sf2.compose(Zero^Zero^Zero^Zero^Zero))
        with profile():
            sampler = CircuitSampler(qi).convert(expectation)
            print(f'sampler: {sampler}')
            print('RESULT:', sampler.eval())
        #with profile():
        #    for _ in range(100):
        #        sampler.eval()

   
if __name__ == "__main__":
    main()
