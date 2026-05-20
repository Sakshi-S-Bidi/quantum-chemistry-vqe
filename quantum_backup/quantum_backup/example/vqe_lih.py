import numpy as np
import time
from mpi4py import MPI
from qulacs import QuantumCircuit, QuantumState, Observable

# -------------------------
# MPI SETUP
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
# REALISTIC LiH Hamiltonian (scaled, no large constant)
# -------------------------
def create_observable():
    obs = Observable(n_qubits)

    # Single qubit terms
    obs.add_operator(0.172, "Z 0")
    obs.add_operator(-0.223, "Z 1")
    obs.add_operator(0.168, "Z 2")
    obs.add_operator(-0.223, "Z 3")

    # Two-qubit interactions
    obs.add_operator(0.120, "Z 0 Z 1")
    obs.add_operator(0.165, "Z 2 Z 3")

    # Entanglement terms
    obs.add_operator(0.045, "X 0 X 1")
    obs.add_operator(0.045, "Y 0 Y 1")
    obs.add_operator(0.045, "X 2 X 3")
    obs.add_operator(0.045, "Y 2 Y 3")

    return obs

# -------------------------
# Ansatz
# -------------------------
def create_circuit(theta):
    circuit = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        circuit.add_RY_gate(i, theta[i])

    for i in range(n_qubits - 1):
        circuit.add_CNOT_gate(i, i + 1)

    for i in range(n_qubits):
        circuit.add_RY_gate(i, theta[i + n_qubits])

    return circuit

# -------------------------
# Energy function
# -------------------------
def compute_energy(theta):
    theta = np.mod(theta, np.pi)

    state = QuantumState(n_qubits)
    observable = create_observable()

    circuit = create_circuit(theta)
    circuit.update_quantum_state(state)

    return observable.get_expectation_value(state)

# -------------------------
# Single VQE run
# -------------------------
def run_vqe():
    best_theta = np.random.rand(2 * n_qubits) * np.pi
    best_energy = compute_energy(best_theta)

    for _ in range(max_iters):
        for j in range(len(best_theta)):

            theta_plus = best_theta.copy()
            theta_plus[j] += step_size
            e_plus = compute_energy(theta_plus)

            theta_minus = best_theta.copy()
            theta_minus[j] -= step_size
            e_minus = compute_energy(theta_minus)

            if e_plus < best_energy:
                best_energy = e_plus
                best_theta = theta_plus
            elif e_minus < best_energy:
                best_energy = e_minus
                best_theta = theta_minus

    return best_energy, best_theta

# -------------------------
# MAIN EXECUTION (MPI SAFE)
# -------------------------
if rank == 0:

    start = time.time()

    print("\n==============================")
    print(" Multi-Start VQE (LiH) ")
    print("==============================")

    num_runs = 5
    results = []

    for i in range(num_runs):
        energy, theta = run_vqe()
        results.append((i, energy))
        print(f"Run {i}: Energy = {energy}")

    best = min(results, key=lambda x: x[1])

    print("\nBest Result:")
    print(f"Run {best[0]} → Energy = {best[1]}")

    end = time.time()
    print("\nExecution Time:", end - start)
