from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler

import math


class CircuitRunner:
    def __init__(self, num_shots: int = 8192, optimization_level: int = 2):
        self.num_shots = num_shots
        self.optimization_level = optimization_level

        self.precision = math.sqrt(1 / self.num_shots)

        self.session = None
        self.estimator = None
        self.sampler = None
        self.pm = None

    def set_session(self, session: Session) -> None:
        self.session = session
        self.estimator = Estimator(mode=self.session)
        self.sampler = Sampler(mode=self.session)
        self.pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=self.session._backend
        )

    def estimate(self, circuit: QuantumCircuit, observables: list[SparsePauliOp]) -> list[float]:

        isa_circuit = self.pm.run(circuit)
        isa_observables = [observable.apply_layout(isa_circuit.layout) for observable in observables]

        job = self.estimator.run([(isa_circuit, isa_observables)], precision=self.precision)
        evs = job.result()[0].data.evs

        return evs

    def sample(self, circuit: QuantumCircuit) -> dict:
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()

        isa_circuit = self.pm.run(measured_circuit)

        job = self.sampler.run([isa_circuit], shots=self.num_shots)
        data_pub = job.result()[0].data
        counts = data_pub.meas.get_counts()

        return counts
