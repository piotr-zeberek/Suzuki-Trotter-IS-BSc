from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .runner import CircuitRunner


class StandardCircuitRunner(CircuitRunner):
    def __init__(self, num_shots: int = 8192, optimization_level: int = 2):
        super().__init__(num_shots, optimization_level)
        
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
