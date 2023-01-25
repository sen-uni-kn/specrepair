from typing import Optional, Sequence, Tuple

import torch

from deep_opt import BoxConstraint, NeuralNetwork, Property
from nn_repair import Counterexample, CounterexampleGenerator


class LinearRegressionVerifier(CounterexampleGenerator):
    """
    This verifier evaluates the property satisfaction function on
    the vertices of the input hyper-rectangle to evaluate property
    satisfaction of a linear model.
    """
    def __init__(self, single_counterexample: bool = False):
        self.single_counterexample = single_counterexample

    def find_counterexample(
        self, target_network: NeuralNetwork, target_property: Property
    ) -> Tuple[Optional[Sequence[Counterexample]], str]:

        if (
            not isinstance(target_network[0], torch.nn.Linear)
            or len(target_network) != 1
        ):
            raise ValueError(f"{target_network} is not a linear regression model. "
                             f"LinearRegressionVerifier only supports linear "
                             f"regression models.")

        output_constraint = target_property.output_constraint
        if not isinstance(output_constraint, BoxConstraint):
            raise ValueError("LinearRegressionVerifier only supports "
                             "properties with BoxConstraints. "
                             f"Got: {output_constraint}.")
        if target_property.input_constraint is not None:
            raise ValueError("LinearRegressionVerifier doesn't support "
                             "properties with input constraints. ")

        input_bounds = target_property.input_bounds(target_network)
        input_vertices = [()]
        for i, (lower, upper) in enumerate(input_bounds):
            input_vertices = [
                bound + (lower,)
                for bound in input_vertices
            ] + [
                bound + (upper,)
                for bound in input_vertices
            ]
        input_vertices = torch.tensor(input_vertices)
        is_satisfied = target_property.property_satisfied(
            input_vertices, target_network
        )

        if is_satisfied.all():
            return [], "VERIFIED"
        else:
            counterexamples = input_vertices[~is_satisfied]
            full_infos = [
                target_property.full_witness(cx.unsqueeze(0), target_network)
                for cx in counterexamples
            ]
            if self.single_counterexample:
                sat_fn_values = [sat_value for _, _, sat_value, _ in full_infos]
                sat_fn_values = torch.stack(sat_fn_values)
                max_violation_index = torch.argmin(sat_fn_values)
                full_infos = [full_infos[max_violation_index]]
            counterexamples = [
                Counterexample(
                    # "inputs" is a batched tensor, remove the batch dimension
                    inputs=inputs[0],
                    network_outputs=outputs,
                    property_satisfaction=sat_val.item(),
                    property=target_property,
                )
                for inputs, outputs, sat_val, _
                in full_infos
            ]
            return counterexamples, "NOT VERIFIED"

    @property
    def name(self) -> str:
        return 'LinearRegressionVerifier'
