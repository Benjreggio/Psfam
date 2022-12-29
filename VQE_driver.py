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

#import numpy as np
import pickle
import time
#from statsmodels.stats.weightstats import DescrStatsW
from math import log
#import logging
#logging.basicConfig(level=logging.INFO)

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
#from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.fake_provider import FakeCasablanca

from qiskit.opflow.primitive_ops import PauliOp

from qiskit.quantum_info.operators import Pauli

from qiskit.tools.visualization import circuit_drawer
from qiskit import QuantumCircuit

#from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
#from qiskit.opflow.operator_base import OperatorBase
#from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp

from rmatrix import random_H
from timing import timing
from Psfam import Pauli_organizer
from q_utils import get_backend

def array_to_Op(Hmat):
    "Convert numpy matrix to qiskit Operator type object."

    N = Hmat.shape[0]
    m = log(N, 2)
    assert m == int(m)
    m = int(m)

    pauli_vec = to_pauli_vec(Hmat)
    #print(pauli_vec)
    #print(len(pauli_vec))
    
    H_op=PauliOp(Pauli('I'*m), 0.0)
    #print(type(H_op))
    for pauli_string in pauli_vec.keys():
        coefficient = pauli_vec[pauli_string]
        #if(abs(coefficient) > 0.0001 ):
        H_op += PauliOp(Pauli(pauli_string), coefficient)
    #print(type(H_op))
    return H_op

def test_ben():
    m = 3
    N = 2**m
    PO = Pauli_organizer(m)
    Hmat = random_H(N)
    evals,evec = np.linalg.eigh(Hmat)
    ref_value=evals[0]
    print("Exact value: ",evals[0])
    
    result = array_to_SummedOp(Hmat, 3)
    print('result:', result)
    print('len(result):', len(result))

def array_to_SummedOp(Hmat):
    "Convert numpy matrix to SummedOp grouped into Pauli-string families."
    N = Hmat.shape[0]
    m = log(N, 2)
    assert m == int(m)
    m = int(m)

    PO = Pauli_organizer(m)
    pauli_vec = to_pauli_vec(Hmat)
    #print(pauli_vec)
    PO.input_pauli_decomps(pauli_vec)
    f = PO.calc_coefficients()
    #for fam in f:
    #    print(fam.to_string())

    H = array_to_Op(Hmat)
    primitive = H.primitive
    #print('primitive:', primitive)
    #print('paulis:', primitive.paulis)
    
    # How stored in Op objects.
    '''
    id_list = \
    ['III', 'IIZ', 'IIX', 'IIY', 'IZI', 'IZZ', 'IZX', 'IZY', 'IXI', 'IXZ', 'IXX', 'IXY', 
    'IYI', 'IYZ', 'IYX', 'IYY', 'ZII', 'ZIZ', 'ZIX', 'ZIY', 'ZZI', 'ZZZ', 'ZZX', 'ZZY', 
    'ZXI', 'ZXZ', 'ZXX', 'ZXY', 'ZYI', 'ZYZ', 'ZYX', 'ZYY', 'XII', 'XIZ', 'XIX', 'XIY', 
    'XZI', 'XZZ', 'XZX', 'XZY', 'XXI', 'XXZ', 'XXX', 'XXY', 'XYI', 'XYZ', 'XYX', 'XYY', 
    'YII', 'YIZ', 'YIX', 'YIY', 'YZI', 'YZZ', 'YZX', 'YZY', 'YXI', 'YXZ', 'YXX', 'YXY', 
    'YYI', 'YYZ', 'YYX', 'YYY']
    '''
    # Will primitive.paulis include the full set irrespective of H?
    id_list = [str(x) for x in primitive.paulis]
    id_dict = { id_list[x]: x for x in range(len(id_list))}
    #print('id_dict:', id_dict)
    
    res = []
    for family in f:
        #print(family.to_string())
        fam_ids = []
        for op in family.to_string():
            fam_ids.append(id_dict[op])
        res.append(fam_ids)
    res[-1].append(0)  # Add the identity operator to the last family.
    #print('res:', res)
    
    result = SummedOp(
        [PauliSumOp(primitive[group], grouping_type="TPB") for group in res],
        coeff=1)

    #print('result:', result)
    #print('len(result):', len(result))

    return result

def run_VQE(H, niters, nshots, backend):
    seed = (int) (10000*np.random.rand()) 
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

def get_Op(Hmat, optype):
    "Qiskit operators from matrix array."
    if optype == 'naive':
        return array_to_Op(Hmat)
    elif optype == 'abelian':
        grouper = AbelianGrouper()
        H = array_to_Op(Hmat)
        return grouper.convert(H)
    elif optype == 'new':
        return array_to_SummedOp(Hmat)
    else:
        msg = f'optype {optype} not recognized [naive, abelian, new].'
        raise ValueError(msg)

def get_vqe_backend(name):
    "Wrapper for obtaining quantum backend."
    if name == 'FakeCasablanca':
        return FakeCasablanca()
    elif name == 'aer_simulator':
        return Aer.get_backend('aer_simulator')
    elif name == "qasm_simulator":
        return get_backend(simulator=True)
    elif name == 'ibmq_quito':
        return get_backend(name='ibmq_quito')
    else:
        msg = f'Backend name {name} not recognized' 
        msg += ' [FakeCasablanca, aer_simulator, qasm_simulator, ibmq_quito]'
        raise ValueError(msg)

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
    main2(5, 'new', 2, 1000, 'ibmq_quito')
    #main2(6, 'naive', 10, 1000, 'qasm_simulator')
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

