from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EvolutionResult:
    expect: List[List[float]]
    counts: List[Dict[str, int]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    gate_counts: List[int] = field(default_factory=list)
    nonlocal_gate_counts: List[int] = field(default_factory=list)
    gate_brakdowns: List[Dict[str, int]] = field(default_factory=list)