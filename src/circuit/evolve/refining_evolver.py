from ...definition import Problem
from ..build import CircuitBuilder
from ..run import CircuitRunner

from .evolver import CircuitEvolver

from .result import EvolutionResult

from typing import Optional

from qiskit.providers import BackendV2 as Backend
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Session

from qiskit.quantum_info import hellinger_fidelity


class RefiningCircuitEvolver(CircuitEvolver):
    def __init__(
        self,
        problem: Problem,
        final_time: float,
        init_dt: float,
        allowed_evs_changes: list[float],
        builder: Optional[CircuitBuilder] = None,
        runner: Optional[CircuitRunner] = None,
        backend: Backend = AerSimulator(),
        sample_circuit: bool = False,
        min_sampling_fidelity: float = 0.95,
        max_steps: int = 1000,
        min_dt: float = 1e-3,
    ):
        super().__init__(
            problem=problem,
            builder=builder,
            runner=runner,
            backend=backend,
            sample_circuit=sample_circuit,
        )
        self.final_time = final_time
        self.init_dt = init_dt
        self.allowed_evs_changes = allowed_evs_changes
        self.min_sampling_fidelity = min_sampling_fidelity
        self.max_steps = max_steps
        self.min_dt = min_dt

    def evolve(self):
        self.reset()
        num_steps = 0
        time = 0.0

        with Session(backend=self.backend) as session:
            self.runner.set_session(session)

            self._collect_results(time)

            while time < self.final_time and num_steps < self.max_steps:
                dt = 2 * min(self.init_dt, self.final_time - time)

                if dt <= 0:
                    break

                refining = True
                H_op = self.problem.hamiltonian_op(time)

                while refining:
                    dt /= 2

                    if dt < self.min_dt:
                        dt = self.min_dt
                        break

                    circuit = self.builder.add_evolution_step(
                        self.builder.evolved_circuit, H_op, dt
                    )

                    refined_circuit = self.builder.add_evolution_step(
                        self.builder.evolved_circuit, H_op, dt / 2
                    )
                    refined_circuit = self.builder.add_evolution_step(
                        refined_circuit,
                        self.problem.hamiltonian_op(time + dt / 2),
                        dt / 2,
                    )

                    E_ops = self.problem.observables_op(time + dt)
                    evs = self.runner.estimate(circuit, E_ops)
                    refined_evs = self.runner.estimate(refined_circuit, E_ops)

                    refining = any(
                        abs(refined - orig) > allowed_change
                        for orig, refined, allowed_change in zip(
                            evs, refined_evs, self.allowed_evs_changes
                        )
                    )

                    if self.sample_circuit:
                        counts = self.runner.sample(circuit)
                        refined_counts = self.runner.sample(refined_circuit)
                        fidelity = hellinger_fidelity(counts, refined_counts)

                        if fidelity < self.min_sampling_fidelity:
                            refining = True

                self.builder.evolve_circuit(H_op, dt)
                time += dt
                num_steps += 1

                if refining:
                    self._collect_results(time)
                else:
                    self._collect_results(time, skip_running=True)

                    for i, ev in enumerate(evs):
                        self.results.expect[i].append(ev)

                    if self.sample_circuit:
                        self.results.counts.append(counts)

        return self.results
