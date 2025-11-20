# Import the qiskit library
import numpy as np
import matplotlib.pylab as plt
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.synthesis import (
    ProductFormula,
    SuzukiTrotter,
    LieTrotter,
)
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit.transpiler.passmanager import StagedPassManager

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    IBMBackend,
)

warnings.filterwarnings("ignore")

from qutip import sigmax, sigmay, sigmaz, basis, SESolver, tensor, qeye


from dataclasses import dataclass

Term = tuple[str, list[int], float]


@dataclass
class QuantumSystemConfig:
    num_qubits: int
    init_state: str
    hamiltonian_tuples: list[Term]
    observables_tuples: list[list[Term]]
    order: int = 2
    reps: int = 1
    synthesis: ProductFormula = None
    num_shots: int = 8192
    optimization_level: int = 2
    backend: Backend | None = AerSimulator()
    pm: StagedPassManager | None = None

    def __post_init__(self):
        self.set_synthesis_parameters(self.order, self.reps)
        self.set_backend(self.backend)

    def set_synthesis_parameters(self, order: int = 2, reps: int = 1):
        self.order = order
        self.reps = reps
        if order == 1:
            self.synthesis = LieTrotter(reps=reps)
        else:
            self.synthesis = SuzukiTrotter(order=order, reps=reps)

    def set_backend(self, backend: Backend | None = None):
        if backend is None:
            backend = AerSimulator()
        self.backend = backend
        self.pm = generate_preset_pass_manager(self.optimization_level, self.backend)


class QuantumSystem:
    def __init__(
        self,
        config: QuantumSystemConfig,
    ):
        self.config = config

        self.__prepare_qiskit_prerequisites()
        self.__prepare_qutip_prerequisites()
        self.clear()

    def __prepare_qiskit_prerequisites(self):
        """Prepare the initial circuit and operators in Qiskit format."""
        self.init_circuit = QuantumCircuit(self.config.num_qubits)
        self.init_circuit.initialize(self.config.init_state)

        self.hamiltonian = SparsePauliOp.from_sparse_list(
            self.config.hamiltonian_tuples, num_qubits=self.config.num_qubits
        )
        self.observables = [
            SparsePauliOp.from_sparse_list(op_tuples, num_qubits=self.config.num_qubits)
            for op_tuples in self.config.observables_tuples
        ]

    def __prepare_qutip_prerequisites(self):
        """Prepare the initial state and operators in QuTiP format based on Qiskit data."""
        self.qutip_init_state = tensor(
            [basis(2, int(bit)) for bit in reversed(self.config.init_state)]
        )
        self.qutip_hamiltonian = self.__convert_qiskit_to_qutip_op(self.hamiltonian)
        self.qutip_observables = [self.__convert_qiskit_to_qutip_op(op) for op in self.observables]

    def __convert_qiskit_to_qutip_op(self, op: SparsePauliOp):
        """Convert a SparsePauliOp to a QuTiP operator."""
        qutip_op = 0
        for pauli_str, coeff in zip(op.paulis.to_labels(), op.coeffs):
            op_list = [qeye(2)] * self.config.num_qubits
            for qubit_idx, pauli_char in enumerate(pauli_str):
                match pauli_char:
                    case "X":
                        op_list[qubit_idx] = sigmax()
                    case "Y":
                        op_list[qubit_idx] = sigmay()
                    case "Z":
                        op_list[qubit_idx] = sigmaz()
            qutip_op += coeff * tensor(op_list)
        return qutip_op

    def clear(self):
        """Clear stored results from previous simulations."""
        self.evolved_state = self.init_circuit.copy()
        self.qiskit_results = [[] for _ in range(len(self.observables))]
        self.qutip_results = None

    def perform_qutip_time_evolution(self, times: np.ndarray | list[float]):
        """Perform exact time evolution using QuTiP's SESolver."""
        solver = SESolver(self.qutip_hamiltonian)
        self.qutip_results = solver.run(
            self.qutip_init_state, times, observables=self.qutip_observables
        ).expect
        return self.qutip_results

    def perform_qiskit_time_evolution(self, times: np.ndarray | list[float]):
        """Perform time evolution using trotterization and Qiskit Estimator."""

        self.clear()
        
        with Session(backend=self.config.backend) as session:
            estimator = Estimator(mode=session)

            self.calculate_expectation_values(estimator)

            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                evolution_gates = PauliEvolutionGate(
                    self.hamiltonian, dt, synthesis=self.config.synthesis
                )
                self.evolved_state.append(evolution_gates, self.evolved_state.qubits)
                self.calculate_expectation_values(estimator)

        return self.qiskit_results

    def calculate_expectation_values(self, estimator: Estimator):
        """Calculate expectation values for all observables given a circuit and store them."""
        precision = np.sqrt(1 / self.config.num_shots)

        isa_circuit = self.config.pm.run(self.evolved_state)
        isa_observables = [observable.apply_layout(isa_circuit.layout) for observable in self.observables]

        job = estimator.run([(isa_circuit, isa_observables)], precision=precision)
        results = job.result()[0].data.evs

        for i, ev in enumerate(results):
            self.qiskit_results[i].append(ev)

    def get_single_step_circuit_data(self):
        """Generate a single-step evolution circuit and return its characteristics."""
        dt = 1.0  # irrelevant (hopefully)
        evolution_gates = PauliEvolutionGate(
            self.hamiltonian, dt, synthesis=self.config.synthesis
        )
        ciruit = QuantumCircuit(self.config.num_qubits)
        ciruit.append(evolution_gates, ciruit.qubits)
        ciruit = ciruit.decompose(reps=3)

        return {
            "depth": ciruit.depth(),
            "gate_count": len(ciruit),
            "nonlocal_gate_count": ciruit.num_nonlocal_gates(),
            "gate_breakdown": {k: v for k, v in ciruit.count_ops().items()},
        }
