from dataclasses import dataclass, field
from typing import List, Dict

from qiskit.quantum_info import DensityMatrix

@dataclass
class EvolutionResult:
    times: List[float] = field(default_factory=list)

    density_matrices: List[DensityMatrix] = field(default_factory=list)
    expect: List[List[float]] = field(default_factory=list)
    counts: List[Dict[str, int]] = field(default_factory=list)

    depths: List[int] = field(default_factory=list)
    gate_counts: List[int] = field(default_factory=list)
    nonlocal_gate_counts: List[int] = field(default_factory=list)
    gate_brakdowns: List[Dict[str, int]] = field(default_factory=list)
