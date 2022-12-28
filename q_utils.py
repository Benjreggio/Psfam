import qiskit
from qiskit import IBMQ

def get_backend(name=False, simulator=True, n_qubits=2):
    provider = IBMQ.load_account()
    if name:
        backend = provider.get_backend(name)
        return backend
    if simulator:
        backend = qiskit.BasicAer.get_backend('qasm_simulator')
        return backend
    else:
        backends = [b for b in provider.backends(simulator=False)
                    if b.configuration().n_qubits >= n_qubits]
        _least_busy = qiskit.providers.ibmq.least_busy
        backend = _least_busy(backends)
        print('Using least busy device:', backend.name())
        return backend