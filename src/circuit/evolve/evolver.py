from ...definition import Problem
from ..build import CircuitBuilder, StandardCircuitBuilder
from ..run import CircuitRunner, StandardCircuitRunner

from .result import EvolutionResult

from abc import ABC, abstractmethod

from typing import Optional

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Session
from qiskit.quantum_info import Statevector, DensityMatrix


class CircuitEvolver(ABC):
    def __init__(
        self,
        problem: Problem,
        builder: Optional[CircuitBuilder] = None,
        runner: Optional[CircuitRunner] = None,
        backend: Backend = AerSimulator(),
        sample_circuit: bool = False,
    ):
        self.problem = problem
        self.builder = builder or StandardCircuitBuilder(
            problem=problem,
        )
        self.runner = runner or StandardCircuitRunner()
        self.backend = backend
        self.sample_circuit = sample_circuit

        self.reset()

    @abstractmethod
    def evolve(self):
        pass

    def reset(self):
        self.builder.reset()
        self.results = EvolutionResult(
            expect=[[] for _ in self.problem.observables_terms],
        )

    def _collect_results(self, time: float, skip_running: bool = False):
        if not skip_running:
            evs = self.runner.estimate(
                self.builder.evolved_circuit, self.problem.observables_op(time)
            )
            for i, ev in enumerate(evs):
                self.results.expect[i].append(ev)

            if self.sample_circuit:
                self.results.counts.append(
                    self.runner.sample(self.builder.evolved_circuit)
                )

        self.results.times.append(time)

        circuit = self.builder.evolved_circuit.decompose(reps=3)

        # self.results.states.append(Statevector.from_instruction(circuit))
        # self.results.depths.append(circuit.depth())
        # self.results.gate_counts.append(len(circuit))
        # self.results.nonlocal_gate_counts.append(circuit.num_nonlocal_gates())
        # self.results.gate_brakdowns.append(
        #     {k.upper(): v for k, v in circuit.count_ops().items()}
        # )
        # self.results.density_matrices.append(
        #     DensityMatrix.from_instruction(circuit)
        # )

        circuit = self.runner.last_transpiled_not_measured_circuit
        if not self.runner.last_transpiled_not_measured_circuit:
            circuit = self.runner.pm.run(self.builder.evolved_circuit)

        circuit = circuit.decompose(reps=3)
        self.results.depths.append(circuit.depth())
        self.results.gate_counts.append(len(circuit))
        self.results.nonlocal_gate_counts.append(circuit.num_nonlocal_gates())
        self.results.gate_brakdowns.append(
            {k.upper(): v for k, v in circuit.count_ops().items()}
        )
        circuit.save_density_matrix()
        self.results.density_matrices.append(
            self.backend.run(circuit).result().data(0)["density_matrix"]
        )