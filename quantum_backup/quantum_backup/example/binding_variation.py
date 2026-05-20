from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import time
from qulacs import QuantumCircuit, QuantumState, Observable

# -------------------------
# Setup
# -------------------------
n_qubits = 8
max_iters = 40
step_size = 0.1

# -------------------------
# Ansatz
# -------------------------
def create_circuit(theta):
    circuit = QuantumCircuit(n_qubits)

    # Layer 1
    for i in range(n_qubits):
        circuit.add_RY_gate(i, theta[i])

    # Entanglement
    for i in range(n_qubits - 1):
        circuit.add_CNOT_gate(i, i + 1)

    # Layer 2
    for i in range(n_qubits):
        circuit.add_RY_gate(i, theta[i + n_qubits])

    return circuit

# -------------------------
# Create Hamiltonian
# -------------------------
def create_observable(interaction_strength):

    observable = Observable(n_qubits)

    # Molecule A
    observable.add_operator(-0.8, "Z 0")
    observable.add_operator(0.2, "Z 1")
    observable.add_operator(0.3, "Z 2")
    observable.add_operator(0.1, "Z 3")

    # Molecule B
    observable.add_operator(-0.8, "Z 4")
    observable.add_operator(0.2, "Z 5")
    observable.add_operator(0.3, "Z 6")
    observable.add_operator(0.1, "Z 7")

    # Interaction terms
    observable.add_operator(interaction_strength, "Z 2 Z 6")
    observable.add_operator(interaction_strength, "Z 3 Z 7")

    return observable

# -------------------------
# Energy Function
# -------------------------
def compute_energy(theta, observable):

    theta = np.mod(theta, np.pi)

    state = QuantumState(n_qubits)

    circuit = create_circuit(theta)
    circuit.update_quantum_state(state)

    return observable.get_expectation_value(state)

# -------------------------
# VQE
# -------------------------
def run_vqe(interaction_strength):



    observable = create_observable(interaction_strength)

    best_theta = np.random.rand(2 * n_qubits) * np.pi
    best_energy = compute_energy(best_theta, observable)

    for _ in range(max_iters):

        for j in range(len(best_theta)):

            theta_plus = best_theta.copy()
            theta_plus[j] += step_size

            e_plus = compute_energy(theta_plus, observable)

            theta_minus = best_theta.copy()
            theta_minus[j] -= step_size

            e_minus = compute_energy(theta_minus, observable)

            if e_plus < best_energy:
                best_energy = e_plus
                best_theta = theta_plus

            elif e_minus < best_energy:
                best_energy = e_minus
                best_theta = theta_minus

    return best_energy

# -------------------------
# Main Experiment
# -------------------------
if rank == 0:
    print("\n==============================")
    print(" Binding Strength Variation ")   
    print("==============================")

    interaction_values = [-0.1, -0.3, -0.5]

    results = []

    start = time.time()

    for strength in interaction_values:

        energy = run_vqe(strength)

        results.append((strength, energy))

        print(f"Interaction Strength {strength} → Energy = {energy}")

# Save results
    with open("binding_results.txt", "w") as f:

     for r in results:
            f.write(f"{r[0]} {r[1]}\n")
    
    end = time.time()

    print("\nExecution Time:", end - start)
    
