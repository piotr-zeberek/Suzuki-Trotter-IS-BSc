from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_addon_utils.slicing import slice_by_gate_types, combine_slices
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.truncating import setup_budget

from .runner import CircuitRunner


class OBPCircuitRunner(CircuitRunner):
    def __init__(
        self,
        num_shots: int = 8192,
        optimization_level: int = 2,
        max_qwc_groups=8,
        max_error_per_slice=None,
    ):
        super().__init__(num_shots, optimization_level)
        self.max_qwc_groups = max_qwc_groups
        self.max_error_per_slice = max_error_per_slice

        self.op_budget = OperatorBudget(max_qwc_groups=self.max_qwc_groups)

        self.truncation_error_budget = (
            None
            if self.max_error_per_slice is None
            else setup_budget(max_error_per_slice=self.max_error_per_slice)
        )

    def estimate(
        self, circuit: QuantumCircuit, observables: list[SparsePauliOp]
    ) -> list[float]:
        # options_dict = {
        #     "default_precision": 0.011,
        #     "resilience_level": 2,
        # }
        # self.estimator._set_options(options_dict)
        
        if circuit.depth() != 0:
            slices = slice_by_gate_types(circuit)
            bp_obs, remaining_slices, metadata = backpropagate(
                observables,
                slices,
                operator_budget=self.op_budget,
                truncation_error_budget=self.truncation_error_budget,
            )
            bp_circuit = combine_slices(remaining_slices)
            
            if bp_circuit is None:
                bp_circuit = QuantumCircuit(circuit.num_qubits)

            print(f"Separated the circuit into {len(slices)} slices.")
            print(f"Backpropagated {metadata.num_backpropagated_slices} slices.")
            print(
                f"New observable has {[len(obs.paulis) for obs in bp_obs]} terms, which can be combined into {[len(obs.group_commuting(qubit_wise=True)) for obs in bp_obs]} groups."
            )
            print(
                f"Note that backpropagating one more slice would result in {metadata.backpropagation_history[-1].num_paulis[0]} terms "
                f"across {metadata.backpropagation_history[-1].num_qwc_groups} groups."
            )
            print("The remaining circuit after backpropagation looks as follows:")
        else:
            bp_obs = observables
            bp_circuit = circuit

            
        isa_circuit = self.pm.run(bp_circuit)
        self.last_transpiled_not_measured_circuit = isa_circuit
        isa_observables = [
            observable.apply_layout(isa_circuit.layout) for observable in bp_obs
        ]

        job = self.estimator.run(
            [(isa_circuit, isa_observables)], precision=self.precision
        )
        evs = job.result()[0].data.evs

        return evs

    def sample(self, circuit: QuantumCircuit) -> dict:
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()

        isa_circuit = self.pm.run(measured_circuit)

        job = self.sampler.run([isa_circuit], shots=self.num_shots)
        data_pub = job.result()[0].data
        counts = data_pub.meas.get_counts()

        return counts