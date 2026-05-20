from qulacs import QuantumCircuit, QuantumState

state = QuantumState(2)
circuit = QuantumCircuit(2)

circuit.add_H_gate(0)
circuit.add_CNOT_gate(0,1)

circuit.update_quantum_state(state)

result = state.sampling(10)

print("Result:", result)
