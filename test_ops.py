#!/usr/bin/env python3
# Andrew Lytle
# Dec 2022

import pickle
import time
#from statsmodels.stats.weightstats import DescrStatsW
from math import log
#import logging
#log = logging.getLogger(__name__)
import cProfile
import pstats

import numpy as np
import scipy.sparse as sp

#from e1plus import get_eigenvalues
from qiskit import Aer
import matplotlib.pyplot as plt
from qiskit.utils import QuantumInstance, algorithm_globals
#from qiskit.circuit.library import TwoLocal,EfficientSU2
#from qiskit.opflow.converters import AbelianGrouper #, NewAbelianGrouper
#from qiskit.providers.fake_provider import FakeCasablanca

from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
#from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import X, Y, Z, I, CX, T, H, S, PrimitiveOp
from qiskit.opflow import (I, X, Y, Z, H, CX, Zero, ListOp, PauliExpectation, 
                           PauliTrotterEvolution, CircuitSampler, MatrixEvolution, 
                           Suzuki)
#from qiskit.circuit import Parameter

from q_utils import get_backend_wrapper as get_backend
from rmatrix import random_H, get_Op
from timing import profile
#from VQE_driver import array_to_Op, array_to_SummedOp
    
def main(m, optype, name, nshots):
    #run_new = False
    #run_abelian = True

    N = 2**m
    Hmat = random_H(N)
    print(Hmat)

    #evals,evec = np.linalg.eigh(Hmat)
    #ref_value=evals[0]
    #print(f"Exact value: {evals[0]}")

    H = get_Op(Hmat, optype)
    print(f"optype: {optype}")
    sf = StateFn(H)
    sf = sf.adjoint()
    state = (Zero^m)  # |0>**m
    _eval = sf.eval(state)
    print(f"Direct evaluation: {_eval}")

    print(f"Using backend {name}")
    backend = get_backend(name)
    qi = QuantumInstance(backend, shots=nshots)
    expectation = PauliExpectation().convert(sf.compose(state))
    sampler = CircuitSampler(qi, attach_results=True).convert(expectation)
    #print(f'sampler: {sampler}')
    print('RESULT:', sampler.eval())

    '''
    if run_new: 
        H = array_to_SummedOp(Hmat, m)
        print(H)
        sf = StateFn(H)
        print('Summed op eval:')
        print(sf.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
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
        print(sf2.adjoint())
        print('abelian eval:')
        print(sf2.adjoint().eval(Zero^Zero^Zero^Zero^Zero))
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
    '''
   
if __name__ == "__main__":
    #main(5, 'new', 'aer_simulator', 1024)
    main(5, 'new', 'FakeCasablanca', 10024)

