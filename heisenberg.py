import numpy as np
import matplotlib.pyplot as plt
 
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.synthesis import LieTrotter
 
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_addon_utils.problem_generators import (
    generate_time_evolution_circuit,
)
from qiskit_addon_utils.slicing import slice_by_gate_types, combine_slices
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.truncating import setup_budget
 
from rustworkx.visualization import graphviz_draw
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions

num_qubits = 10
layout = [(i - 1, i) for i in range(1, num_qubits)]
 
# Instantiate a CouplingMap object
coupling_map = CouplingMap(layout)
# graphviz_draw(coupling_map.graph, method="circo")

# Get a qubit operator describing the Heisenberg XYZ model
hamiltonian = generate_xyz_hamiltonian(
    coupling_map,
    coupling_constants=(np.pi / 8, np.pi / 4, np.pi / 2),
    ext_magnetic_field=(np.pi / 3, np.pi / 6, np.pi / 9),
)

from src.definition import Problem, Term
hamiltonian_tuples = [
    Term(pauli_str, qubits, coeff)
    for pauli_str, qubits, coeff in hamiltonian.to_sparse_list()
]

observables_tuples = [
    hamiltonian_tuples,
    [Term("Z", [i], 1 / num_qubits) for i in range(num_qubits)],
]

problem = Problem(
    num_qubits=num_qubits,
    init_state="1" * num_qubits,
    hamiltonian_terms=hamiltonian_tuples,
    observables_terms=observables_tuples,
)

from src.circuit import OBPCircuitRunner
from src.circuit import StandardCircuitBuilder
from src.circuit import FixedCircuitEvolver

builder = StandardCircuitBuilder(problem=problem, synthesis_order=1)

runner = OBPCircuitRunner(
    num_shots=8192,
    optimization_level=2,
    max_qwc_groups=8,
    # max_error_per_slice=1e-2,
)

import numpy as np

evolver = FixedCircuitEvolver(
    problem=problem, 
    times=np.linspace(0, 10, 21),
    builder=builder,
    runner=runner,
)

qiskit_results = evolver.evolve()

from src.solver import QutipSolver

qutip_solver = QutipSolver(problem, qiskit_results.times)
qutip_results = qutip_solver.solve()

plt.figure(figsize=(10, 6))
labels = ["Energy", "Magnetization",]
for i in range(len(observables_tuples)):
    plt.subplot(3, 1, i + 1)
    plt.plot(
        qiskit_results.times,
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
plt.savefig("heisenberg.png")
# plt.savefig("fixed_ising_dynamics_with_OBP_4th_order.png")


# runner.set_session(