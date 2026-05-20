from qulacs import QuantumCircuit, QuantumState
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# quantum state
num_qubits = 2
state = QuantumState(num_qubits, use_multi_cpu=True)

# circuit
circuit = QuantumCircuit(num_qubits)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

# apply circuit
circuit.update_quantum_state(state)

# sampling
result = state.sampling(10)

# print only once
if rank == 0:
    print("Result:", result)
