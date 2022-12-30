# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:23:17 2022

@author: ntbutt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:57:44 2021

@author: noumanbutt
"""

import pickle
import time
#from statsmodels.stats.weightstats import DescrStatsW
from math import log
#import logging
#logging.basicConfig(level=logging.INFO)

import cProfile
import pstats
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA,COBYLA
from qiskit.circuit.library import TwoLocal,EfficientSU2
#from qiskit.circuit.library import RealAmplitudes

#from qiskit.tools.visualization import circuit_drawer
from qiskit import QuantumCircuit

from rmatrix import get_Op
from timing import timing
from q_utils import get_backend_wrapper as get_vqe_backend

# in main() only [deprecated]
from decompose_pauli import to_pauli_vec  
from rmatrix import random_H, array_to_Op, array_to_SummedOp
from qiskit.opflow.converters import AbelianGrouper #, NewAbelianGrouper
from qiskit import Aer

def run_VQE(H, niters, nshots, backend):
    seed = 42
    iterations = niters
    algorithm_globals.random_seed = seed
    #backend = Aer.get_backend('aer_simulator')
    #backend = FakeCasablanca()
    qi = QuantumInstance(backend=backend, shots=nshots, seed_simulator=seed, seed_transpiler=seed)

    counts = []
    values = []
    devs = []
        
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        devs.append(std)

    ansatz =  TwoLocal(3,rotation_blocks='ry', entanglement_blocks='cz',entanglement='full',reps=2)
    #ansatz = EfficientSU2(3, su2_gates=['ry', 'x'], entanglement='full', reps=2)
    #circuit_drawer(ansatz, output='mpl', style={'backgroundcolor': '#EEEEEE'})
    #print(ansatz.num_parameters)
    #ansatz =  TwoLocal(rotation_blocks='ry', entanglement_blocks='cz',entanglement='full',reps=2)
 
    #spsa = SPSA(maxiter=iterations,learning_rate=0.02,perturbation=0.3)
 
    spsa = COBYLA(maxiter=iterations,rhobeg=0.2)

    vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result,quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=H)

    return result

def main2(m, optype, niters, nshots, name):
    N = 2**m
    Hmat = random_H(N)
    
    evals, evec = np.linalg.eigh(Hmat)
    ref_value=evals[0]
    print(f"Exact value: {evals[0]}")
    
    H = get_Op(Hmat, optype)
    backend = get_vqe_backend(name)
    print(f'#### VQE using optype: {optype}, using backend: {name}  ####')
    print('type(H):', type(H))
    print('# of families:', len(H))
    with timing():
        result = run_VQE(H, niters, nshots, backend)
    #H = array_to_Op(Hmat)
    print(f'VQE on aer simulator (no noise): {result.eigenvalue.real:.5f}')
    #print(f'Std Dev of VQE value: {devs[-1]:.5f}')
    print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')
    

if __name__ == "__main__":
    #main2(5, 'new', 2, 1000, 'ibmq_quito')
    main2(3, 'abelian', 10, 1000, 'ibmq_quito')
    #print()
    #main2(6, 'naive', 10, 1000, 'aer_simulator')
    #print()
    #main2(3, 'naive', 10, 1000, 'FakeCasablanca')
    #main2(6, 'abelian', 10, 1000)
    #print()
    #main2(6, 'new', 10, 1000)

    #test_ben()

def main():
    iters=2
    m = 3
    N = 2**m
    #S = sp.random(8, 8, density=0.9) # sparse random hermitian matrix
    #S = 0.5*( S + S.T)
    #Hmat = S.A
    Hmat = random_H(N)

    evals,evec = np.linalg.eigh(Hmat)
    ref_value=evals[0]
    print("Exact value: ",evals[0])
    pauli_vec = to_pauli_vec(Hmat)

    print("# of Pauli-Strings: ",len(pauli_vec))

     # of Is = log_2(M)
    
    H = array_to_Op(Hmat)
    
    #print("# of Pauli-Strings: ",H)
    t1=time.time()    
    #grouper = NewAbelianGrouper()
    grouper = AbelianGrouper()
 
    #H = grouper.convert(H)    
 
    t2=time.time()

    print("Time taken to find families: ",t2-t1)   
    
    for n in range(iters):
        seed = (int) (10000*np.random.rand()) 
        iterations = 200
        algorithm_globals.random_seed = seed
        backend = Aer.get_backend('aer_simulator')
        qi = QuantumInstance(backend=backend, shots=10000, seed_simulator=seed, seed_transpiler=seed)

        counts = []
        values = []
        devs = []
        
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
            devs.append(std)

        ansatz =  TwoLocal(3,rotation_blocks='ry', entanglement_blocks='cz',entanglement='full',reps=2)
        #ansatz = EfficientSU2(3, su2_gates=['ry', 'x'], entanglement='full', reps=2)
        #circuit_drawer(ansatz, output='mpl', style={'backgroundcolor': '#EEEEEE'})
        #print(ansatz.num_parameters)
        #ansatz =  TwoLocal(rotation_blocks='ry', entanglement_blocks='cz',entanglement='full',reps=2)
 
 
        #spsa = SPSA(maxiter=iterations,learning_rate=0.02,perturbation=0.3)
 
        spsa = COBYLA(maxiter=iterations,rhobeg=0.2)

        time0 = time.time()
        if(n==1):
            H = grouper.convert(H)  
        print("# of families: ",len(H))
        vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result,quantum_instance=qi)
        result = vqe.compute_minimum_eigenvalue(operator=H)
        #E0_vqe.append(result.eigenvalue.real)
        #opt_params.append(result.optimal_point)
        #err.append(devs[-1])
  
        #print(vqe._get_eigenstate())
        time1 = time.time()
   
        print("Time taken to run VQE: ", time1-time0)
        print(f'VQE on aer simulator (no noise): {result.eigenvalue.real:.5f}')
        print(f'Std Dev of VQE value: {devs[-1]:.5f}')
        print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')

