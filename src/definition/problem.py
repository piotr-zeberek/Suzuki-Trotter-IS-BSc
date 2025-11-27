from typing import Optional, List
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

from .term import Term


class Problem:
    def __init__(
        self,
        num_qubits: int,
        init_state: Optional[str]|QuantumCircuit,
        hamiltonian_terms: List[Term],
        observables_terms: List[List[Term]],
    ):
        self.num_qubits = num_qubits
        self.init_state = init_state
        self.hamiltonian_terms = hamiltonian_terms
        self.observables_terms = observables_terms

        self.td_hamiltonian_terms: List[Term] = []
        self.td_observables_terms: List[List[Term]] = []

        self.ti_hamiltonian_op: SparsePauliOp = None
        self.ti_observables_op: List[SparsePauliOp] = []

        self._separate_time_dependency()

    def _separate_time_dependency(self):
        ti_hamiltonian_tuples: list[Term] = []
        for term in self.hamiltonian_terms:
            if callable(term.coefficient):
                self.td_hamiltonian_terms.append(term)
            else:
                ti_hamiltonian_tuples.append(term)

        self.ti_hamiltonian_op = SparsePauliOp.from_sparse_list(
            ti_hamiltonian_tuples, num_qubits=self.num_qubits
        )

        for observable_terms in self.observables_terms:
            ti_observable_tuples: List[Term] = []
            td_observable_terms: List[Term] = []

            for term in observable_terms:
                if callable(term.coefficient):
                    td_observable_terms.append(term)
                else:
                    ti_observable_tuples.append(term)

            self.td_observables_terms.append(td_observable_terms)

            self.ti_observables_op.append(
                SparsePauliOp.from_sparse_list(
                    ti_observable_tuples, num_qubits=self.num_qubits
                )
            )

    def hamiltonian_op(self, time: float = 0.0) -> SparsePauliOp:
        if not self.td_hamiltonian_terms:
            return self.ti_hamiltonian_op

        return self.ti_hamiltonian_op + SparsePauliOp.from_sparse_list(
            [term(time) for term in self.td_hamiltonian_terms],
            num_qubits=self.num_qubits,
        )

    def observables_op(self, time: float = 0.0) -> List[SparsePauliOp]:
        if not any(self.td_observables_terms):
            return self.ti_observables_op

        return self.ti_observables_op + [
            SparsePauliOp.from_sparse_list(
                [term(time) for term in td_observable_terms],
                num_qubits=self.num_qubits,
            )
            for td_observable_terms in self.td_observables_terms
        ]
