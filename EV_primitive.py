
""" Building up measurements to be used in expectation values.
    Based on codes found at
    https://nonhermitian.org/posts/2021/2021-10-13-expval_program.html
    https://nonhermitian.org/posts/2021/2021-10-07-vqe_program.html

    Andrew Lytle
    Dec 2022
"""

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeSantiago
from qiskit.providers.ibmq.runtime import UserMessenger


from timing import profile

def main():
    msg = UserMessenger()
    backend = FakeSantiago()

    # |0000> + |1111>
    #qc = QuantumCircuit(16)
    qc = QuantumCircuit(4)
    qc.h(2)
    qc.cx(2, 1)
    qc.cx(1, 0)
    qc.cx(2, 3)
    qc.measure_all()
    #with profile():
    #    for _ in range(10):
    #        result = backend.run(qc, shots=1024).result()
    #raw_counts = result.get_counts()
    #print(f'raw counts: {raw_counts}')
    print(opstr_to_meas_circ('XXIY'))

def opstr_to_meas_circ(op_str):
    """Takes a list of operator strings and makes circuit with the correct post-rotations for measurements.

    Parameters:
        op_str (list): List of strings representing the operators needed for measurements.

    Returns:
        list: List of circuits for measurement post-rotations
    """
    num_qubits = len(op_str[0])
    circs = []
    for op in op_str:
        qc = QuantumCircuit(num_qubits)
        for idx, item in enumerate(op):
            if item == 'X':
                qc.h(num_qubits-idx-1)
            elif item == 'Y':
                qc.sdg(num_qubits-idx-1)
                qc.h(num_qubits-idx-1)
        circs.append(qc)
    return circs

if __name__ == "__main__":
    main()
