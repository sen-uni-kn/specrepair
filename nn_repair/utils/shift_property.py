from typing import Any, Tuple

import torch
from ruamel.yaml import yaml_object
from deep_opt.models.property import Property, OutputConstraint, property_yaml_context_manager


@yaml_object(property_yaml_context_manager)
class ShiftedOutputConstraint(OutputConstraint):
    """
    Shifts the satisfaction function of another output constraint by a fixed offset.

    Used for handling spurious counterexamples that have an internal violation value recorded from the
    counterexample generator that returned that spurious counterexamples.

    Spurious counterexamples may come up due to the error introduced by serialising a neural network.
    This output constraint allows to handle such issues.
    """
    def __init__(self, base_constraint: OutputConstraint, offset: float):
        self.original = base_constraint
        self.offset = offset

    def satisfaction_function(self, network_inputs: torch.Tensor, network_outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        original_satisfaction, original_satisfied = \
            self.original.satisfaction_function(network_inputs, network_outputs)
        shifted_violation = original_satisfaction - self.offset
        shifted_satisfied = torch.where(
            (original_satisfaction >= 0) == original_satisfied,
            shifted_violation >= 0, shifted_violation > 0
        )
        return shifted_violation, shifted_satisfied

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ShiftedOutputConstraint):
            return False
        return self.original == o.original and self.offset == o.offset

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.original, self.offset))

    def __repr__(self):
        return f"{self.original} shifted by {self.offset:.4e}"

    yaml_tag = '!ShiftedOutputConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(ShiftedOutputConstraint.yaml_tag, {
            'base_output_constraint': data.original,
            'offset': data.offset
        })

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        if len(mapping) != 2:
            raise ValueError(f"Malformed ShiftedOutputConstraint: wrong number of attributes: {mapping}")
        return ShiftedOutputConstraint(mapping['base_output_constraint'], mapping['offset'])


def shift_property(prop: Property, offset: float) -> Property:
    return Property(
        lower_bounds=prop.lower_bounds,
        upper_bounds=prop.upper_bounds,
        output_constraint=ShiftedOutputConstraint(prop.output_constraint, offset),
        input_constraint=prop.input_constraint,
        property_name=prop.property_name + f' shifted by {offset:.2e}'
    )
