from .definition import Problem

from .circuit.build import CircuitBuilder
from .circuit.run import CircuitRunner
from .circuit.evolve import FinalCircuitEvolver

from typing import Optional
from qiskit_aer import AerSimulator

from qiskit.providers import BackendV2 as Backend

from qiskit_addon_mpf.costs import (
    setup_exact_problem,
    setup_sum_of_squares_problem,
)
from qiskit_addon_mpf.static import setup_static_lse

import numpy as np


class MPF:
    def __init__(
        self,
        problem: Problem,
        final_time: float,
        steps: list[int],
        builder: Optional[CircuitBuilder] = None,
        runner: Optional[CircuitRunner] = None,
        backend: Backend = AerSimulator(),
    ):
        self.evolvers = [
            FinalCircuitEvolver(
                problem=problem,
                final_time=final_time,
                num_steps=num_steps,
                builder=builder,
                runner=runner,
                backend=backend,
            )
            for num_steps in steps
        ]

        self.steps = steps

        self.lse = setup_static_lse(
            self.steps, order=self.evolvers[0].builder.synthesis_order, symmetric=True
        )
        self.model_exact, self.coeffs_exact = setup_exact_problem(self.lse)
        self.model_approx, self.coeffs_approx = setup_sum_of_squares_problem(
            self.lse, max_l1_norm=3.0
        )
        
        self.model_exact.solve()
        self.model_approx.solve()
        
        self.evolvers_results = None
        self.mpf_exact_results = None
        self.mpf_approx_results = None
        
    def run(self):
        self.evolvers_results = np.array([sum(evolver.evolve().expect, start=[]) for evolver in self.evolvers])
        
        self.mpf_exact_results = self.evolvers_results.T @ self.coeffs_exact.value
        self.mpf_approx_results = self.evolvers_results.T @ self.coeffs_approx.value
        
        return self.evolvers_results, self.mpf_exact_results, self.mpf_approx_results
