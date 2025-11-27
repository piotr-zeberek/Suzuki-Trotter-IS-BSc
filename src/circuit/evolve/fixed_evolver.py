from ...definition import Problem
from ..build import CircuitBuilder
from ..run import CircuitRunner

from .evolver import CircuitEvolver

from .result import EvolutionResult

from typing import Optional

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Session


class FixedCircuitEvolver(CircuitEvolver):
    def __init__(
        self,
        problem: Problem,
        times: list[float],
        builder: Optional[CircuitBuilder] = None,
        runner: Optional[CircuitRunner] = None,
        backend: Backend = AerSimulator(),
        sample_circuit: bool = False,
    ):
        super().__init__(
            problem=problem,
            builder=builder,
            runner=runner,
            backend=backend,
            sample_circuit=sample_circuit,
        )
        self.times = times
        self.dts = [
            self.times[i + 1] - self.times[i] for i in range(len(self.times) - 1)
        ]

    def evolve(self):
        self.reset()

        with Session(backend=self.backend) as session:
            self.runner.set_session(session)

            self._collect_results(self.times[0])

            for time, dt in zip(self.times[:-1], self.dts):
                self.builder.evolve_circuit(self.problem.hamiltonian_op(time), dt)
                self._collect_results(time + dt)

        return self.results