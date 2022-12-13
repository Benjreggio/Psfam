#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
import itertools, functools
#from e1plus import get_eigenvalues

op_I = sp.eye(2)
op_Z = sp.dia_matrix([[1,0],[0,-1]])
op_X = sp.dia_matrix([[0,1],[1,0]])
op_Y = -1j * op_Z @ op_X

C1 = [[1,0],[0,0]]
C2 = [[0,1],[0,0]]
C3 = [[0,0],[1,0]]
C4 = [[0,0],[0,1]]



C_ops = { "C1" : C1, "C2" :C2, "C3": C3, "C4" : C4 }
#C_ops = { "C1" : C1, "C2" :C2, "C4" : C4 }
pauli_ops = { "I" : op_I, "Z" : op_Z, "X": op_X, "Y" : op_Y }

# kronecker (tensor) product of two sparse matrices
# returns a sparse matrix in "dictionary of keys" format (to eliminate zeros)
def sp_kron_dok(mat_A, mat_B): return sp.kron(mat_A, mat_B, format = "dok")

def to_pauli_vec(mat):
    pauli_vec = {} # the dictionary we are saving

    mat_vec = np.array(mat).ravel()
    num_qubits = int(np.log2(np.sqrt(mat_vec.size)))

    for pauli_string in itertools.product(pauli_ops.keys(), repeat = num_qubits):
        # construct this pauli string as a matrix
        ops = [ pauli_ops[tag] for tag in pauli_string ]
        op = functools.reduce(sp_kron_dok, ops)

        # compute an inner product, same as tr(A @ B) but faster
        op_vec = op.reshape((1,4**num_qubits))
        coefficient = ( op_vec * mat_vec ).sum() / 2**num_qubits
        if coefficient != 0:
            pauli_vec["".join(pauli_string)] = coefficient

    return pauli_vec

def to_C_vec(mat):
    C_vec = {} # the dictionary we are saving

    mat_vec = np.array(mat).ravel()
    num_qubits = int(np.log2(np.sqrt(mat_vec.size)))

    for C_string in itertools.product(C_ops.keys(), repeat = num_qubits):
        # construct this pauli string as a matrix
        ops = [ C_ops[tag] for tag in C_string ]
        op = functools.reduce(sp_kron_dok, ops)

        # compute an inner product, same as tr(A @ B) but faster
        op_vec = op.reshape((1,4**num_qubits))
        coefficient = ( op_vec * mat_vec ).sum() / 2**num_qubits
        if coefficient != 0:
            C_vec["".join(C_string)] = coefficient

    return C_vec


#mat = [[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,1,0,0]] # for example...

#print(to_pauli_vec(mat))

'''

num_qubits=2
for C_string in itertools.product(C_ops.keys(), repeat = num_qubits):
    
    ops = [ C_ops[tag] for tag in C_string ]
    op = functools.reduce(sp_kron_dok, ops)
    

    print("".join(C_string))
    
    C = op.todense()
    
    if(np.allclose(C,np.tril(C)) == True):
        print(C)
        print(to_pauli_vec(C))
    
    print("------------------------------")


#S = sp.random(32, 32, density=0.125) # sparse random hermitian matrix
#S = S + S.T
mat = [[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,1,0,0]] # for example...
#print(S.A)
#print(np.count_nonzero(S.A))
#print(len(to_pauli_vec(S.A)))
'''
'''
lmax=20
nmax=10
flag=0
M=8
gs=[0.8]#,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.2,2.4,2.6]

for g in gs:
    
 H,evals=get_eigenvalues(g,M,lmax,nmax) 

 print("# of non-zero elements in H: ", np.count_nonzero(H)) 

 pauli_vec = to_pauli_vec(H)
 print(pauli_vec)
 print("# of all Pauli strings: ",len(pauli_vec))
 n=0
 for pauli_string in pauli_vec.keys():
    coefficient = pauli_vec[pauli_string]
    
    if(abs(coefficient) < 0.0001):
     
     n += 1
     
     
 print("# of negligible co-efficients: ", n)
 print("# of non-zero Pauli strings: ", len(pauli_vec)- n) 
''' 
