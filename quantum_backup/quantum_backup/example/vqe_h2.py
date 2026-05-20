import numpy as np
from qulacs import QuantumCircuit, QuantumState, Observable
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_qubits = 2

# Hamiltonian
observable = Observable(n_qubits)
observable.add_operator(-1.05, "Z 0")
observable.add_operator(0.39, "Z 1")
observable.add_operator(0.18, "X 0 X 1")
observable.add_operator(-0.01, "Z 0 Z 1")

# Ansatz
def create_circuit(theta):
    circuit = QuantumCircuit(n_qubits)

    # Layer 1
    circuit.add_RY_gate(0, theta[0])
    circuit.add_RY_gate(1, theta[1])
    circuit.add_CNOT_gate(0, 1)

    # Layer 2 (SAFE version)
    circuit.add_RY_gate(0, theta[2])
    circuit.add_RY_gate(1, theta[3])
    circuit.add_CNOT_gate(0, 1)   # same direction (IMPORTANT)

    return circuit

# Energy function
def compute_energy(theta):
    # keep values in safe range
    theta = np.mod(theta, np.pi)

    state = QuantumState(n_qubits)
    circuit = create_circuit(theta)
    circuit.update_quantum_state(state)

    return observable.get_expectation_value(state)

# Random search optimizer
best_theta = np.random.rand(4) * np.pi
best_energy = compute_energy(best_theta)

energy_history = []

for i in range(50):
    new_theta = best_theta + (np.random.rand(4) - 0.5) * 0.3
    new_energy = compute_energy(new_theta)

    if new_energy < best_energy:
        best_energy = new_energy
        best_theta = new_theta

    energy_history.append(best_energy)

# Output
if rank == 0:
    print("Final Energy:", best_energy)
    print("Best Theta:", best_theta)

    # Save convergence to file
    with open("energy_log.txt", "w") as f:
        for val in energy_history:
            f.write(str(val) + "\n")

    print("Energy log saved to file")
