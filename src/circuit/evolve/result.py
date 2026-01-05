from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

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

    def write_to_files(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # times
        with open(f"{directory}/times.dat", "w") as f:
            for time in self.times:
                f.write(f"{time}\n")
        
        # expect
        with open(f"{directory}/expect.dat", "w") as f:
            for expect_values in zip(*self.expect):
                expect_str = " ".join(f"{val}" for val in expect_values)
                f.write(f"{expect_str}\n")

        # counts
        if self.counts:
            with open(f"{directory}/counts.dat", "w") as f:
                for count_dict in self.counts:
                    count_str = " ".join(f"{k}:{v}" for k, v in count_dict.items())
                    f.write(f"{count_str}\n")

        # depths
        with open(f"{directory}/depths.dat", "w") as f:
            for depth in self.depths:
                f.write(f"{depth}\n")

        # gate_counts
        with open(f"{directory}/gate_counts.dat", "w") as f:
            for gate_count in self.gate_counts:
                f.write(f"{gate_count}\n")

        # nonlocal_gate_counts
        with open(f"{directory}/nonlocal_gate_counts.dat", "w") as f:
            for nonlocal_gate_count in self.nonlocal_gate_counts:
                f.write(f"{nonlocal_gate_count}\n")

        # gate_brakdowns
        with open(f"{directory}/gate_brakdowns.dat", "w") as f:
            for breakdown in self.gate_brakdowns:
                breakdown_str = " ".join(f"{k}:{v}" for k, v in breakdown.items())
                f.write(f"{breakdown_str}\n")