from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print("Running VQE...")

    # ALL your existing code here
import numpy as np
import time
from qulacs import QuantumCircuit, QuantumState, Observable

# -------------------------
# Setup
# -------------------------
n_qubits = 8  # 2 molecules (4 qubits each)

start = time.time()

# -------------------------
# Define Hamiltonian
# -------------------------
observable = Observable(n_qubits)

# ---- Molecule A (qubits 0–3) ----
observable.add_operator(-0.8, "Z 0")
observable.add_operator(0.2, "Z 1")
observable.add_operator(0.3, "Z 2")
observable.add_operator(0.1, "Z 3")

# ---- Molecule B (qubits 4–7) ----
observable.add_operator(-0.8, "Z 4")
observable.add_operator(0.2, "Z 5")
observable.add_operator(0.3, "Z 6")
observable.add_operator(0.1, "Z 7")

# ---- Interaction terms (IMPORTANT 🔥) ----
observable.add_operator(-0.2, "Z 2 Z 6")
observable.add_operator(-0.2, "Z 3 Z 7")

# -------------------------
# Ansatz (stable multi-layer)
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
# Energy Function
# -------------------------
def compute_energy(theta):
    theta = np.mod(theta, np.pi)

    state = QuantumState(n_qubits)
    circuit = create_circuit(theta)
    circuit.update_quantum_state(state)

    return observable.get_expectation_value(state)

# -------------------------
# Optimizer (Coordinate Descent)
# -------------------------
best_theta = np.random.rand(2 * n_qubits) * np.pi
best_energy = compute_energy(best_theta)

energy_history = []

max_iters = 50
step_size = 0.2

for iteration in range(max_iters):
    for j in range(len(best_theta)):
        theta_plus = best_theta.copy()
        theta_plus[j] += step_size
        energy_plus = compute_energy(theta_plus)

        theta_minus = best_theta.copy()
        theta_minus[j] -= step_size
        energy_minus = compute_energy(theta_minus)

        if energy_plus < best_energy:
            best_energy = energy_plus
            best_theta = theta_plus

        elif energy_minus < best_energy:
            best_energy = energy_minus
            best_theta = theta_minus

    energy_history.append(best_energy)

# -------------------------
# Output
# -------------------------
print("\n==============================")
print(" Interaction Simulation ")
print("==============================")

print("Final Energy (with interaction):", best_energy)

# Save log
with open("interaction_energy_log.txt", "w") as f:
    for val in energy_history:
        f.write(str(val) + "\n")

end = time.time()
print("Execution Time:", end - start)
