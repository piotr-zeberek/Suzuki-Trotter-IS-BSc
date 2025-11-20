from typing import Optional

from qiskit import QuantumCircuit
from qiskit.synthesis import (
    SuzukiTrotter,
    LieTrotter,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

class CircuitBuilder:
    def __init__(
        self,
        num_qubits: int,
        init_state: Optional[str],
        synthesis_order: int = 2,
        synthesis_reps: int = 1,
    ):
        self.num_qubits = num_qubits
        self.init_state = init_state
        self.synthesis_order = synthesis_order
        self.synthesis_reps = synthesis_reps

        self.init_circuit = QuantumCircuit(num_qubits)

        if init_state is not None:
            self.init_circuit.initialize(init_state)

        self.evolved_circuit = self.init_circuit.copy()

        if synthesis_order == 1:
            self.synthesis = LieTrotter(reps=synthesis_reps)
        elif synthesis_order >= 2:
            self.synthesis = SuzukiTrotter(
                order=(
                    synthesis_order if synthesis_order % 2 == 0 else synthesis_order - 1
                ),
                reps=synthesis_reps,
            )

    def evolve_circuit(self, hamiltonian_op: SparsePauliOp, dt: float):
        evolution_gates = PauliEvolutionGate(
            hamiltonian_op,
            time=dt,
            synthesis=self.synthesis,
        )

        self.evolved_circuit.append(evolution_gates, self.evolved_circuit.qubits)

        return self.evolved_circuit

    def reset(self):
        self.evolved_circuit = self.init_circuit.copy()
