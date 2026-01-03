from qutip import (
    sigmax,
    sigmay,
    sigmaz,
    basis,
    SESolver,
    tensor,
    qeye,
    Qobj,
    QobjEvo,
    coefficient,
    ket
)

from ..definition import Term, Problem
from .base import Solver


class QutipSolver(Solver):
    def __init__(self, problem: Problem, times: list[float]):
        super().__init__(problem)
        self.times = times

        self._convert_to_qutip()

        self.result = None

    def solve(self):
        """Perform exact time evolution using QuTiP's SESolver."""
        solver = SESolver(self.hamiltonian)
        solver.options['store_states'] = True
        self.results = solver.run(
            self.init_state, self.times, e_ops=self.observables
        )
        return self.results

    def _convert_to_qutip(self) -> None:
        """Convert problem data to QuTiP objects."""
        self.init_state = ket(self.problem.init_state)
        self.hamiltonian = sum(
            self._Term_to_Qobj(term) for term in self.problem.hamiltonian_terms
        )
        self.observables = [
            sum(self._Term_to_Qobj(term) for term in observable_terms)
            for observable_terms in self.problem.observables_terms
        ]

    def _Term_to_Qobj(self, op: Term) -> Qobj | QobjEvo:
        """Convert a Term to a Qutip Qobj or QobjEvo."""

        op_list = [qeye(2)] * self.problem.num_qubits
        for pauli_char, qubit_idx in zip(op.pauli_string, op.qubit_indices):
            match pauli_char:
                case "X":
                    op_list[qubit_idx] = sigmax()
                case "Y":
                    op_list[qubit_idx] = sigmay()
                case "Z":
                    op_list[qubit_idx] = sigmaz()

        coeff = (
            coefficient(op.coefficient) if callable(op.coefficient) else op.coefficient
        )
        qutip_op = coeff * tensor(reversed(op_list))
        # qutip_op = coeff * tensor(op_list)

        return qutip_op
