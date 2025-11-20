from src.definition import Problem, Term
from src.solver import QutipSolver

from src.control import FixedTimeControl
from src.circuit import CircuitEvolver

import numpy as np
import matplotlib.pyplot as plt

# num_qubits = 6
# init_ones = [2, 3]
# init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
# J = 0.2
# h = 1.2
# alpha = np.pi / 8.0
# time = 30.0
# qutip_times = np.linspace(0, time, 301)
# times = np.linspace(0, time, 61)


# def ising_hamiltonian_tuples(nqubits, J, h, alpha):
#     # List of Hamiltonian terms as 3-tuples containing
#     # (1) the Pauli string,
#     # (2) the qubit indices corresponding to the Pauli string,
#     # (3) the coefficient.
#     ZZ_tuples = [Term("ZZ", [i, i + 1], -J) for i in range(0, nqubits - 1)]
#     Z_tuples = [Term("Z", [i], -h * np.sin(alpha)) for i in range(0, nqubits)]
#     X_tuples = [Term("X", [i], -h * np.cos(alpha)) for i in range(0, nqubits)]

#     # We create the Hamiltonian as a SparsePauliOp, via the method
#     # `from_sparse_list`, and multiply by the interaction term.
#     return [*ZZ_tuples, *Z_tuples, *X_tuples]


# hamiltonian_tuples = ising_hamiltonian_tuples(num_qubits, J, h, alpha)

# magnetization_tuples = [Term("Z", [i], 1.0 / num_qubits) for i in range(0, num_qubits)]
# correlation_tuples = [
#     Term("ZZ", [i, i + 1], 1.0 / (num_qubits - 1)) for i in range(0, num_qubits - 1)
# ]
# observables_tuples = [hamiltonian_tuples, magnetization_tuples, correlation_tuples]


# problem = Problem(
#     num_qubits=num_qubits,
#     init_state=init_state,
#     hamiltonian_terms=hamiltonian_tuples,
#     observables_terms=observables_tuples,
# )

# qutip_solver = QutipSolver(problem, qutip_times)
# qutip_results = qutip_solver.solve()

# control = FixedTimeControl(times=times)
# evolver = CircuitEvolver(problem, control)

# qiskit_results = evolver.evolve()

# plt.figure(figsize=(10, 6))
# labels = ["Energy", "Magnetization", "Nearest-neighbor Correlation"]
# for i in range(len(observables_tuples)):
#     plt.subplot(3, 1, i + 1)
#     plt.plot(
#         qutip_times,
#         qutip_results.expect[i],
#         label="QuTiP Exact",
#         color="blue",
#     )
#     plt.plot(
#         times,
#         qiskit_results.expect[i],
#         "o-",
#         label="Qiskit Trotterized",
#         color="orange",
#     )
#     plt.title(labels[i])
#     plt.xlabel("Time")
#     plt.ylabel("Expectation Value")
#     plt.grid()
#     plt.legend()
# plt.tight_layout()
# plt.savefig("ising_dynamics.png")


num_qubits = 1
init_ones = []
init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
w = 1
w0 = 1
w1 = 1
time = 10.0
qutip_times = np.linspace(0, time, 501)
times = np.linspace(0, time, 51)


def rabi_oscillation_hamiltonian(w, w0, w1):
    # List of Hamiltonian terms as 3-tuples containing
    # (1) the Pauli string,
    # (2) the qubit indices corresponding to the Pauli string,
    # (3) the coefficient.
    X_tuples = [
        Term("X", [i], lambda t: -0.5 * w1 * np.cos(w * t))
        for i in range(0, num_qubits)
    ]
    Y_tuples = [
        Term("Y", [i], lambda t: 0.5 * w1 * np.sin(w * t)) for i in range(0, num_qubits)
    ]
    Z_tuples = [Term("Z", [i], -0.5 * w0) for i in range(0, num_qubits)]

    # We create the Hamiltonian as a SparsePauliOp, via the method
    # `from_sparse_list`, and multiply by the interaction term.
    return [*X_tuples, *Y_tuples, *Z_tuples]


hamiltonian_tuples = rabi_oscillation_hamiltonian(w, w0, w1)

X_tuples = [Term("X", [i], 1.0) for i in range(0, num_qubits)]
Y_tuples = [Term("Y", [i], 1.0) for i in range(0, num_qubits)]
Z_tuples = [Term("Z", [i], 1.0) for i in range(0, num_qubits)]
I0_tuples = [Term("I", [0], 0.5), Term("Z", [0], 0.5)]
I1_tuples = [Term("I", [0], 0.5), Term("Z", [0], -0.5)]
observables_tuples = [X_tuples, Y_tuples, Z_tuples, I0_tuples, I1_tuples]

problem = Problem(
    num_qubits=num_qubits,
    init_state=init_state,
    hamiltonian_terms=hamiltonian_tuples,
    observables_terms=observables_tuples,
)

qutip_solver = QutipSolver(problem, qutip_times)
qutip_results = qutip_solver.solve()

control = FixedTimeControl(times=times)
evolver = CircuitEvolver(problem, control)
qiskit_results = evolver.evolve()

labels = ["X Expectation", "Y Expectation", "Z Expectation", "P(0)", "P(1)"]
for i in range(len(observables_tuples)):
    plt.plot(qutip_times, qutip_results.expect[i], label=labels[i])
    plt.plot(
        times,
        qiskit_results.expect[i],
        "o-",
    )
    lines = plt.gca().get_lines()
    lines[-1].set_color(lines[-2].get_color())

plt.legend()
plt.tight_layout()
plt.savefig("rabi_dynamics.png")
