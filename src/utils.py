from .circuit.evolve.result import EvolutionResult
from qutip import Result

from qiskit.quantum_info import state_fidelity, DensityMatrix


def calc_fidelities(
    approx_results: EvolutionResult,
    exact_results: EvolutionResult | Result,
) -> list[float]:
    if isinstance(exact_results, EvolutionResult):
        fidelities = [
            state_fidelity(approx_dm, exact_dm)
            for approx_dm, exact_dm in zip(
                approx_results.density_matrices, exact_results.density_matrices
            )
        ]
    elif isinstance(exact_results, Result):
        fidelities = [
            state_fidelity(
                approx_dm,
                DensityMatrix(exact_state.full().flatten()),
            )
            for approx_dm, exact_state in zip(
                approx_results.density_matrices, exact_results.states
            )
        ]
    else:
        raise TypeError("exact_results must be of type EvolveResult or QuTiP Result")
    return fidelities
