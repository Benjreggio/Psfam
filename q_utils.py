import qiskit
from qiskit import IBMQ
from qiskit import Aer
from qiskit.providers.fake_provider import FakeCasablanca

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

def get_backend_wrapper(name):
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