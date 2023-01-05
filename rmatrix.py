#!/usr/bin/env python3
# Andrew Lytle
# Nov 2022

from math import log
from collections import defaultdict

import numpy as np
from numpy.linalg import eig
from numpy.random import normal
from scipy.stats import unitary_group

from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info.operators import Pauli
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.converters import AbelianGrouper
#from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
#from qiskit.opflow.state_fns.state_fn import StateFn

from decompose_pauli import to_pauli_vec
from Psfam import Pauli_organizer

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

def test_random_H():
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

def get_groups(m):
    """Specification of Pauli string families suitable for use in Qiskit.
    
    Args: 
        int m: Number of qubits
    
    Returns:
        defaultdict<list> {1: [i, j, k,...], 2: [l, m, n,...], ...}
        where [i, j, k,...] are integers specifying the Pauli operators
        in the family, according to the internal Qiskit ordering
        (e.g. id_list below.)

    Note: Currently function generates a random H matrix to obtain 
    the default qiskit operator ordering (H.primitive). This
    can probably be rewritten so that this isn't necessary.
    """
    N = 2**m
    PO = Pauli_organizer(m)

    # Long way to get primitive elements..
    Hmat = random_H(N)
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
    for family in PO.f:
        #print(family.to_string())
        fam_ids = []
        for op in family.to_string():
            fam_ids.append(id_dict[op])
        res.append(fam_ids)
    res[-1].append(0)  # Add the identity operator to the last family.

    groups = defaultdict(list)
    groups = {i: res[i] for i in range(len(res))}
    return groups

def array_to_SummedOp(Hmat):
    "Convert numpy matrix to SummedOp grouped into Pauli-string families."
    N = Hmat.shape[0]
    m = log(N, 2)
    assert m == int(m)
    m = int(m)
    #PO = Pauli_organizer(m)

    H = array_to_Op(Hmat)
    primitive = H.primitive
    #print('primitive:', primitive)
    #print('paulis:', primitive.paulis)
    
    groups = get_groups(m)

    result = SummedOp(
        [PauliSumOp(primitive[group], grouping_type="TPB") for group in groups.values()],
        coeff=1)

    #print('result:', result)
    #print('len(result):', len(result))

    return result

def get_Op(Hmat, optype):
    "Qiskit operators from matrix array."
    if optype == 'naive':
        return array_to_Op(Hmat)
    elif optype == 'abelian':
        grouper = AbelianGrouper()
        H = array_to_Op(Hmat)
        return grouper.convert(H)
        '''       
        print(f"type(H): {type(H)}")
        print(f"H.primitive: {H.primitive}")
        grouped = grouper.convert(H.primitive)
        H = StateFn(grouped, is_measurement=True, coeff=H.coeff)
        cob = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
        return cob.convert(H).reduce()
        '''
    elif optype == 'new':
        return array_to_SummedOp(Hmat)
    else:
        msg = f'optype {optype} not recognized [naive, abelian, new].'
        raise ValueError(msg)

if __name__ == '__main__':
    #test_random_H()
    print(get_groups(2))
    Hmat = random_H(4)
    print(array_to_SummedOp(Hmat))
    
