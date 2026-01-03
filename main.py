from src.definition import Problem, Term
from src.solver import QutipSolver

from src.circuit import FixedCircuitEvolver, RefiningCircuitEvolver
from src.circuit import OBPCircuitRunner
from src.circuit import StandardCircuitBuilder

import numpy as np
import matplotlib.pyplot as plt

from qiskit_ibm_runtime.fake_provider import FakeBrisbane as backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit.quantum_info import DensityMatrix, state_fidelity

## Ising chain with transverse field

num_qubits = 6
init_ones = [0, 1]
init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
J = 0.2
h = 1.2
alpha = np.pi / 8.0
time = 1.0
steps = 21
times = np.linspace(0, time, steps)


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


# runner = OBPCircuitRunner()

builder = StandardCircuitBuilder(problem=problem, synthesis_order=2)

noise_model = NoiseModel.from_backend(backend())
simulator = AerSimulator(noise_model=noise_model)

evolver = FixedCircuitEvolver(
    problem=problem,
    times=times,
    # backend=simulator,
    # runner=runner,
    builder=builder,
)

# evolver = RefiningCircuitEvolver(
#     problem=problem,
#     final_time=time,
#     builder=builder,
#     backend=simulator,
#     init_dt=(times[1] - times[0]),
#     allowed_evs_changes=[1e-2, 1e-1, 1e-1],
#     # sample_circuit=True,
#     # min_sampling_fidelity=0.95,
# )

qiskit_results = evolver.evolve()

c = 1
qutip_times = c * qiskit_results.times
qutip_solver = QutipSolver(problem, qutip_times)
qutip_results = qutip_solver.solve()

plt.figure(figsize=(10, 6))
labels = ["Energy", "Magnetization", "Nearest-neighbor Correlation"]
for i in range(len(observables_tuples)):
    plt.subplot(3, 1, i + 1)
    plt.plot(
        qutip_times,
        qutip_results.expect[i],
        label="QuTiP Exact",
        color="blue",
    )
    plt.plot(
        qiskit_results.times,
        qiskit_results.expect[i],
        "o-",
        label="Qiskit Trotterized",
        color="orange",
    )
    plt.title(labels[i])
    plt.xlabel("Time")
    plt.ylabel("Expectation Value")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig("ising_dynamics_test.png")

fidelities = []

for qiskit_dm, qutip_state in zip(
    qiskit_results.density_matrices,
    qutip_results.states,
    # qiskit_results.states, qutip_results.states
):
    qutip_dm = DensityMatrix(qutip_state.full().flatten())
    fidelity = state_fidelity(qiskit_dm, qutip_dm)
    fidelities.append(fidelity)


plt.figure(figsize=(8, 4))
plt.plot(range(len(fidelities)), fidelities, "o-")
plt.title("Fidelity between Qiskit and QuTiP States")
plt.xlabel("Step")
plt.ylabel("Fidelity")
plt.grid()
plt.tight_layout()
plt.ylim(0.0, 1.05)
plt.axhline(y=1.0, color="gray", linestyle="--", label="Ideal Fidelity")
plt.savefig("ising_fidelity.png")

# num_qubits = 1
# init_ones = []
# init_state = "".join(["1" if i in init_ones else "0" for i in range(num_qubits)])[::-1]
# w = 1
# w0 = 1
# w1 = 1
# time = 10.0
# qutip_times = np.linspace(0, time, 51)
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

# # builder = StandardCircuitBuilder(problem=problem, synthesis_order=2)
# # runner = OBPCircuitRunner(
# #     num_shots=8192,
# #     optimization_level=2,
# #     max_qwc_groups=8,
# #     # max_error_per_slice=1e-2,
# # )

# evolver = FixedCircuitEvolver(
#     problem=problem,
#     times=times,
#     # runner=runner,
#     # builder=builder,
# )

# # evolver = RefiningCircuitEvolver(
# #     problem=problem,
# #     final_time=time,
# #     init_dt=times[1] - times[0],
# #     allowed_evs_changes=[1e-1 for _ in observables_tuples],
# #     # runner=runner,
# #     # builder=builder,
# #     min_dt= 0.01,
# #     sample_circuit=True,
# # )

# qiskit_results = evolver.evolve()

# # labels = ["X Expectation", "Y Expectation", "Z Expectation", "P(0)", "P(1)"]
# # for i in range(len(observables_tuples)):
# #     plt.plot(qutip_times, qutip_results.expect[i], label=labels[i])
# #     plt.plot(
# #         qiskit_results.times,
# #         qiskit_results.expect[i],
# #         "o-",
# #     )
# #     lines = plt.gca().get_lines()
# #     lines[-1].set_color(lines[-2].get_color())

# # plt.legend()
# # plt.tight_layout()
# # # plt.savefig("fixed_rabi_dynamics.png")
# # plt.savefig("refined_rabi_dynamics.png")

# # fidelities = []

# # for qiskit_state, qutip_state in zip(
# #     qiskit_results.states, qutip_results.states
# # ):
# #     qiskit_array = qiskit_state.data
# #     qutip_array = qutip_state.full().flatten()
# #     fidelity = np.abs(np.vdot(qiskit_array, qutip_array)) ** 2
# #     fidelities.append(fidelity)

# # plt.figure(figsize=(8, 4))
# # plt.plot(qiskit_results.times, fidelities, "o-")
# # plt.title("Fidelity between Qiskit and QuTiP States")
# # plt.xlabel("Time")
# # plt.ylabel("Fidelity")
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig("rabi_fidelity.png")
