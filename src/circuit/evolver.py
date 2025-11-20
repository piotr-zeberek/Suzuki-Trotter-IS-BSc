from ..control import TimeControl
from ..definition import Problem

from .builder import CircuitBuilder
from .runner import CircuitRunner

from typing import Optional

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Session


class EvolutionResult:
    def __init__(self, expect: list[list[float]], counts: Optional[list[dict]] = None):
        self.expect = expect
        self.counts = counts


class CircuitEvolver:
    def __init__(
        self,
        problem: Problem,
        control: TimeControl,
        builder: Optional[CircuitBuilder] = None,
        runner: CircuitRunner = CircuitRunner(),
        backend: Backend = AerSimulator(),
        sample_circuit: bool = False,
    ):
        self.problem = problem
        self.control = control
        self.builder = builder or CircuitBuilder(
            num_qubits=problem.num_qubits,
            init_state=problem.init_state,
        )
        self.runner = runner
        self.backend = backend
        self.sample_circuit = sample_circuit

        self.sampler_results = []
        self.estimator_results = [[] for _ in self.problem.observables_terms]

    def evolve(self):
        self.builder.reset()

        with Session(backend=self.backend) as session:
            self.runner.set_session(session)

            self._collect_results(self.control.time())

            allowed, dt = self.control.advance()

            while allowed:
                self.builder.evolve_circuit(self.problem.hamiltonian_op(self.control.time()), dt)
                self._collect_results(self.control.time())
                allowed, dt = self.control.advance()

        return EvolutionResult(
            expect=self.estimator_results,
            counts=self.sampler_results if self.sample_circuit else None,
        )

    def _collect_results(self, time):
        evs = self.runner.estimate(self.builder.evolved_circuit, self.problem.observables_op(time))
        for i, ev in enumerate(evs):
            self.estimator_results[i].append(ev)

        if self.sample_circuit:
            self.sampler_results.append(
                self.runner.sample(self.builder.evolved_circuit)
            )
