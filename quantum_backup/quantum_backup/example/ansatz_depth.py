import numpy as np
import time
from mpi4py import MPI
from qulacs import QuantumCircuit, QuantumState, Observable

# -------------------------
# MPI Setup
# -------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# -------------------------
# Setup
# -------------------------
n_qubits = 4
max_iters = 40
step_size = 0.1

np.random.seed(42)

# -------------------------
# Hamiltonian
# -------------------------
observable = Observable(n_qubits)

observable.add_operator(-1.0, "Z 0")
observable.add_operator(0.5, "Z 1")
observable.add_operator(0.3, "Z 2")
observable.add_operator(0.2, "Z 3")

observable.add_operator(0.2, "Z 0 Z 1")
observable.add_operator(0.1, "Z 2 Z 3")

# -------------------------
# Create Ansatz
# -------------------------
def create_circuit(theta, depth):

    circuit = QuantumCircuit(n_qubits)

    theta_index = 0

    for d in range(depth):

        # Rotation layer
        for i in range(n_qubits):
            circuit.add_RY_gate(i, theta[theta_index])
            theta_index += 1

        # Entanglement layer
        for i in range(n_qubits - 1):
            circuit.add_CNOT_gate(i, i + 1)

    return circuit

# -------------------------
# Energy Function
# -------------------------
def compute_energy(theta, depth):

    theta = np.mod(theta, np.pi)

    state = QuantumState(n_qubits)

    circuit = create_circuit(theta, depth)

    circuit.update_quantum_state(state)

    return observable.get_expectation_value(state)

# -------------------------
# VQE
# -------------------------
def run_vqe(depth):

    num_params = depth * n_qubits

    best_theta = np.random.rand(num_params) * np.pi

    best_energy = compute_energy(best_theta, depth)

    for _ in range(max_iters):

        for j in range(len(best_theta)):

            theta_plus = best_theta.copy()
            theta_plus[j] += step_size

            e_plus = compute_energy(theta_plus, depth)

            theta_minus = best_theta.copy()
            theta_minus[j] -= step_size

            e_minus = compute_energy(theta_minus, depth)

            if e_plus < best_energy:
                best_energy = e_plus
                best_theta = theta_plus

            elif e_minus < best_energy:
                best_energy = e_minus
                best_theta = theta_minus

    return best_energy

# -------------------------
# MAIN EXECUTION (MPI SAFE)
# -------------------------
if rank == 0:

    print("\n==============================")
    print(" Ansatz Depth Comparison ")
    print("==============================")

    depths = [1, 2, 3]

    results = []

    start = time.time()

    for depth in depths:

        energy = run_vqe(depth)

        results.append((depth, energy))

        print(f"Depth {depth} → Energy = {energy}")

    # Save results
    with open("ansatz_depth_results.txt", "w") as f:

        for r in results:
            f.write(f"{r[0]} {r[1]}\n")

    end = time.time()

    print("\nExecution Time:", end - start)
