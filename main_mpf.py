from src.definition import Problem, Term
from src.solver import QutipSolver

from src.circuit import FixedCircuitEvolver, RefiningCircuitEvolver
from src.circuit import OBPCircuitRunner
from src.circuit import StandardCircuitBuilder

import numpy as np
import matplotlib.pyplot as plt
from src.mpf import MPF


num_qubits = 6
init_ones = [2, 3]
init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
J = 0.2
h = 1.2
alpha = np.pi / 8.0
time = 30.0
qutip_times = np.linspace(0, time, 301)
times = np.linspace(0, time, 61)


def ising_hamiltonian_tuples(nqubits, J, h, alpha):
    # List of Hamiltonian terms as 3-tuples containing
    # (1) the Pauli string,
    # (2) the qubit indices corresponding to the Pauli string,
    # (3) the coefficient.
    ZZ_tuples = [Term("ZZ", [i, i + 1], -J) for i in range(0, nqubits - 1)]
    Z_tuples = [Term("Z", [i], -h * np.sin(alpha)) for i in range(0, nqubits)]
    X_tuples = [Term("X", [i], -h * np.cos(alpha)) for i in range(0, nqubits)]

    # We create the Hamiltonian as a SparsePauliOp, via the method
    # `from_sparse_list`, and multiply by the interaction term.
    return [*ZZ_tuples, *Z_tuples, *X_tuples]


hamiltonian_tuples = ising_hamiltonian_tuples(num_qubits, J, h, alpha)

magnetization_tuples = [Term("Z", [i], 1.0 / num_qubits) for i in range(0, num_qubits)]
correlation_tuples = [
    Term("ZZ", [i, i + 1], 1.0 / (num_qubits - 1)) for i in range(0, num_qubits - 1)
]
observables_tuples = [hamiltonian_tuples, magnetization_tuples, correlation_tuples]


problem = Problem(
    num_qubits=num_qubits,
    init_state=init_state,
    hamiltonian_terms=hamiltonian_tuples,
    observables_terms=observables_tuples,
)

qutip_solver = QutipSolver(problem, qutip_times)
qutip_results = qutip_solver.solve()

# runner = OBPCircuitRunner()

builder = StandardCircuitBuilder(problem=problem, synthesis_order=2)

# evolver = FixedCircuitEvolver(
#     problem=problem,
#     times=times,
#     runner=runner,
#     builder=builder,
# )


mpf = MPF(
    problem=problem,
    final_time=time,
    steps=[20, 50, 80],
    builder=builder,
    # runner=runner,
)

pfs_results, mpf_exact_results, mpf_approx_results = mpf.run()

labels = ["Hamiltonian Expectation", "Magnetization Expectation", "Correlation Expectation"]
for label, exact, pf_result, mpf_exact, mpf_approx in zip(
    labels,
    [evs[-1] for evs in qutip_results.expect],
    pfs_results[-1],
    mpf_exact_results,
    mpf_approx_results,
):
    print(f"=== {label} ===")
    print("QuTiP Exact:", exact)
    print("PF Result:", pf_result)
    print("MPF Exact Approximation:", mpf_exact)
    print("MPF Sum-of-Squares Approximation:", mpf_approx)
    print()


# num_qubits = 1
# init_ones = []
# init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
# w = 1
# w0 = 1
# w1 = 1
# time = 10.0
# qutip_times = np.linspace(0, time, 501)
# times = np.linspace(0, time, 51)


# def rabi_oscillation_hamiltonian(w, w0, w1):
#     # List of Hamiltonian terms as 3-tuples containing
#     # (1) the Pauli string,
#     # (2) the qubit indices corresponding to the Pauli string,
#     # (3) the coefficient.
#     X_tuples = [
#         Term("X", [i], lambda t: -0.5 * w1 * np.cos(w * t))
#         for i in range(0, num_qubits)
#     ]
#     Y_tuples = [
#         Term("Y", [i], lambda t: 0.5 * w1 * np.sin(w * t)) for i in range(0, num_qubits)
#     ]
#     Z_tuples = [Term("Z", [i], -0.5 * w0) for i in range(0, num_qubits)]

#     # We create the Hamiltonian as a SparsePauliOp, via the method
#     # `from_sparse_list`, and multiply by the interaction term.
#     return [*X_tuples, *Y_tuples, *Z_tuples]


# hamiltonian_tuples = rabi_oscillation_hamiltonian(w, w0, w1)

# X_tuples = [Term("X", [i], 1.0) for i in range(0, num_qubits)]
# Y_tuples = [Term("Y", [i], 1.0) for i in range(0, num_qubits)]
# Z_tuples = [Term("Z", [i], 1.0) for i in range(0, num_qubits)]
# I0_tuples = [Term("I", [0], 0.5), Term("Z", [0], 0.5)]
# I1_tuples = [Term("I", [0], 0.5), Term("Z", [0], -0.5)]
# observables_tuples = [X_tuples, Y_tuples, Z_tuples, I0_tuples, I1_tuples]

# problem = Problem(
#     num_qubits=num_qubits,
#     init_state=init_state,
#     hamiltonian_terms=hamiltonian_tuples,
#     observables_terms=observables_tuples,
# )

# qutip_solver = QutipSolver(problem, qutip_times)
# qutip_results = qutip_solver.solve()

# builder = StandardCircuitBuilder(problem=problem, synthesis_order=2)
# # runner = OBPCircuitRunner(
# #     num_shots=8192,
# #     optimization_level=2,
# #     max_qwc_groups=8,
# #     # max_error_per_slice=1e-2,
# # )

# mpf = MPF(
#     problem=problem,
#     final_time=time,
#     steps=[10, 20, 30, 40, 50],
#     builder=builder,
#     # runner=runner,
# )

# pfs_results, mpf_exact_results, mpf_approx_results = mpf.run()

# labels = ["X Expectation", "Y Expectation", "Z Expectation", "P(0)", "P(1)"]
# for label, exact, pf_result, mpf_exact, mpf_approx in zip(
#     labels,
#     [evs[-1] for evs in qutip_results.expect],
#     pfs_results[-1],
#     mpf_exact_results,
#     mpf_approx_results,
# ):
#     print(f"=== {label} ===")
#     print("QuTiP Exact:", exact)
#     print("PF Result:", pf_result)
#     print("MPF Exact Approximation:", mpf_exact)
#     print("MPF Sum-of-Squares Approximation:", mpf_approx)
#     print()
    
    