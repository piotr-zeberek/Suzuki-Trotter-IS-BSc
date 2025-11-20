from typing import Optional, List, Callable, NamedTuple


class Term(NamedTuple):
    pauli_string: str
    qubit_indices: List[int]
    coefficient: float | Callable[[float], float]

    def __call__(self, time: float = 0.0) -> tuple[str, List[int], float]:
        if callable(self.coefficient):
            coeff = self.coefficient(time)
        else:
            coeff = self.coefficient
        return (self.pauli_string, self.qubit_indices, coeff)
