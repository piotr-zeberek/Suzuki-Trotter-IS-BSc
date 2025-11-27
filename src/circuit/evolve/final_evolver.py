from ...definition import Problem
from ..build import CircuitBuilder
from ..run import CircuitRunner

from .evolver import CircuitEvolver

from typing import Optional

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Session


class FinalCircuitEvolver(CircuitEvolver):
    def __init__(
        self,
        problem: Problem,
        final_time: float,
        num_steps: int,
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
        self.final_time = final_time
        self.num_steps = num_steps
        self.dt = final_time / num_steps

    def evolve(self):
        self.reset()

        with Session(backend=self.backend) as session:
            self.runner.set_session(session)

            for step in range(self.num_steps + 1):
                time = step * self.dt
                self.builder.evolve_circuit(self.problem.hamiltonian_op(time), self.dt)
                
            self._collect_results(self.final_time)

        return self.results
