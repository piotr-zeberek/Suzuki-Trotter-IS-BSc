from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler

import math


class CircuitRunner(ABC):
    def __init__(self, num_shots: int = 8192, optimization_level: int = 3):
        self.num_shots = num_shots
        self.optimization_level = optimization_level

        self.precision = math.sqrt(1 / self.num_shots)

        self.session = None
        self.estimator = None
        self.sampler = None
        self.pm = None
        
        self.last_transpiled_not_measured_circuit = None

    def set_session(self, session: Session) -> None:
        self.session = session
        self.estimator = Estimator(mode=self.session)
        self.sampler = Sampler(mode=self.session)
        self.pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=self.session._backend
        )

    def run(
        self, circuit: QuantumCircuit, observables: list[SparsePauliOp]
    ) -> tuple[list[float], dict]:
        return self.estimate(circuit, observables), self.sample(circuit)

    @abstractmethod
    def estimate(
        self, circuit: QuantumCircuit, observables: list[SparsePauliOp]
    ) -> list[float]:
        pass

    @abstractmethod
    def sample(self, circuit: QuantumCircuit) -> dict:
        pass
