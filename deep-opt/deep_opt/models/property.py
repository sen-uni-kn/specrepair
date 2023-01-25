import itertools
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Dict, Union, Optional, Iterator, Any
from ruamel.yaml import YAML, yaml_object

import numpy as np
import torch

from deep_opt.models.neural_network import NeuralNetwork

# using pure=True removes a quirk regarding octal numbers in YAML (but probably increases runtime)
property_yaml_context_manager = YAML(typ='safe', pure=True)


# ======================================================================================================================
# ======================================================================================================================
#                                               Output Constraints
# ======================================================================================================================
# ======================================================================================================================

class OutputConstraint(ABC):
    """
    Abstract base class for constraints on network outputs, with single or multiple variables.
    Implementing classes need to provide a satisfaction_function.
    More details can be found in Property.
    """

    @abstractmethod
    def satisfaction_function(self, outputs, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class SingleVarOutputConstraint(OutputConstraint):
    """
    Abstract base class for constraints on network outputs (for a single input variable).
    Implementing classes need to provide a satisfaction_function.
    More details can be found in Property.
    """

    @abstractmethod
    def satisfaction_function(self, outputs: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a 1d tensor (vector) measuring the satisfaction or violation of this output constraint for the given
        batch of network outputs.

        This function is designed to support calculating satisfactions for multiple independent
        network outputs for different network inputs at once (vectorization).
        For this purpose, the network outputs are always presented in batches.
        Also, if ``outputs`` contains the outputs for only one set of inputs, ``outputs`` is at least two-dimensional
        with the first dimension being the batch dimension. This batch dimension will contain only a single element
        if the satisfaction for one set of outputs is to be calculated.
        The different elements of the batch need to be treated individually. It may not influence the satisfaction
        for one particular set of outputs, what other outputs there are in the batch it is processed with.
        However, the batch should be processed in bulk, in a vectorized fashion whenever possible to allow faster
        processing.

        This function is minimized (for each element of a batch individually) to obtain violations.

        The second return value of this method is a 1-d tensor (vector) of booleans that specify
        whether the constraint is satisfied for the given network outputs in the batch individually.

        Implementations should support automatic gradient calculation with pytorch.

        :param outputs: A batch of outputs this SingleVarOutputConstraint will operate on.
         The first dimension is the batch dimension. This dimension needs to be present.
        :param inputs: The batch of inputs that produced the outputs.
         The first dimension is the batch dimension. This dimension needs to be present.
        :return: A vector measuring the satisfaction of this property for each element
         of the batch of network outputs individually. Values closer to zero correspond to the property being more close
         to being violated. Negative values indicate violation of the property.
         Zero may indicate violation or satisfaction. Whether the first return value indicates satisfaction or violation
         is encoded in the second output. The second output is a vector of booleans that states whether this
         output constraint is satisfied for the network outputs in the given batch individually.
        """
        raise NotImplementedError()


class MultiVarOutputConstraint(OutputConstraint):
    """
    Abstract base class for constraints on multiple network outputs (for multiple input variables).
    Implementing classes need to provide a satisfaction_function.
    More details can be found in MultiInputProperty.
    """

    @abstractmethod
    def satisfaction_function(self, outputs: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the violation/satisfaction of this output constraint
        for the multiple network outputs.
        The second return value indicates whether the output constraint is violated.

        For more detail, refer to OutputConstraint.

        :param outputs: Several batches of network outputs for different inputs that this
         MultiVarOutputConstraint will operate on.
         Every element of `outputs` needs to have a batch dimension as first dimension.
         This dimension is mandatory even for a batch of one.
         All batches need to have the same number of elements.
        :param inputs: The batches of inputs that produced the outputs.
         These batches similarly need to have a batch dimension as first dimension
         and batch-lengths need to agree.
        :return: A vector measuring the satisfaction of this property for each element
         of the batch of network outputs individually.
         Values closer to zero correspond to the property being more close
         to being violated.
         Negative values indicate violation of the property.
         Zero may indicate violation or satisfaction.
         Whether the first return value indicates satisfaction or violation
         is encoded in the second output.
         The second output is a vector of booleans that states whether this
         output constraint is satisfied for the network outputs in the given batch individually.
        """
        raise NotImplementedError()


@yaml_object(property_yaml_context_manager)
class ConstraintOr(OutputConstraint):
    def __init__(self, *output_constraints: OutputConstraint):
        self.constraints = tuple(output_constraints)

    def satisfaction_function(self, network_outputs, network_inputs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        results = (constr.satisfaction_function(network_outputs, network_inputs) for constr in self.constraints)
        values, satisfied = zip(*results)
        # values are vectors. vstack creates a matrix concatenating the vectors as row vectors.
        # max(..., dim=0) then removes the row dimension again and yields a vector output
        # where max has been calculated for each column (batch element) individually
        values = torch.max(torch.vstack(values), dim=0).values
        satisfied = torch.any(torch.vstack(satisfied), dim=0)
        return values, satisfied

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ConstraintOr):
            return False
        return self.constraints == o.constraints

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.constraints)

    def __repr__(self):
        return f"OR{self.constraints}"

    yaml_tag = '!ConstraintOr'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_sequence(ConstraintOr.yaml_tag, data.constraints)

    @classmethod
    def from_yaml(cls, loader, node):
        constraints = loader.construct_sequence(node, deep=True)
        return ConstraintOr(*constraints)


@yaml_object(property_yaml_context_manager)
class ConstraintAnd(OutputConstraint):
    def __init__(self, *output_constraints: OutputConstraint):
        self.constraints = tuple(output_constraints)

    def satisfaction_function(self, network_outputs, network_inputs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        results = (constr.satisfaction_function(network_outputs, network_inputs) for constr in self.constraints)
        values, satisfied = zip(*results)
        values = torch.min(torch.vstack(values), dim=0).values
        satisfied = torch.all(torch.vstack(satisfied), dim=0)
        return values, satisfied

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ConstraintAnd):
            return False
        return self.constraints == o.constraints

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.constraints)

    def __repr__(self):
        return f"AND{self.constraints}"

    yaml_tag = '!ConstraintAnd'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_sequence(ConstraintAnd.yaml_tag, data.constraints)

    @classmethod
    def from_yaml(cls, loader, node):
        constraints = loader.construct_sequence(node, deep=True)
        return ConstraintAnd(*constraints)


@yaml_object(property_yaml_context_manager)
class BoxConstraint(SingleVarOutputConstraint):
    def __init__(self, output_index: int, operator: str, bound: float):
        """
        Create an output constraint in the form

        .. math::
            out_i > r
        or

        .. math::
            out_i < r
        or

        .. math::
            out_i >= r
        or

        .. math::
            out_i <= r

        :param output_index: The index of the target output feature of a network (i).
        :param operator: Supported values: less than: '<', 'lt';
         greater than: '>', 'gt'; less than or equals: '<=', '=<', 'le';  greater than or equals: '>=', '=>', 'ge'.
        :param bound: The bound on the output value (r).
        """
        if operator != '<' and operator != '>' and operator != "lt" and operator != "gt" \
                and operator != '<=' and operator != '=<' and operator != 'le' \
                and operator != '>=' and operator != '=>' and operator != 'ge':
            raise ValueError("Unknown operator: " + operator + ". Allowed: <, >, lt, gt, <=, =<, le, >=, =>, ge.")
        self.output_index = int(output_index)
        self.less_than = operator == "<" or operator == "lt" or operator == "<=" or operator == "=<" or operator == "le"
        self.exclude_equals = operator == "<" or operator == ">" or operator == "lt" or operator == "gt"
        self.bound = float(bound)

    def satisfaction_function(self, network_outputs: torch.Tensor, network_inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        value: torch.Tensor
        if self.less_than:  # also <=
            value = self.bound - network_outputs[:, self.output_index]
        else:  # greater than (or equals)
            value = network_outputs[:, self.output_index] - self.bound
        satisfied: torch.Tensor
        if self.exclude_equals:  # < and >
            satisfied = torch.gt(value, 0)
        else:  # >= and =<
            satisfied = torch.ge(value, 0)
        return value, satisfied

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, BoxConstraint):
            return False
        return self.output_index == o.output_index and self.less_than == o.less_than \
               and self.exclude_equals == o.exclude_equals and self.bound == o.bound

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.output_index, self.less_than, self.exclude_equals, self.bound))

    def __repr__(self):
        operator_str: str
        if self.less_than and self.exclude_equals:
            operator_str = '<'
        elif self.less_than and not self.exclude_equals:
            operator_str = '=<'
        elif not self.less_than and self.exclude_equals:
            operator_str = '>'
        else:
            operator_str = '>='
        return f'out{self.output_index} {operator_str} {self.bound}'

    yaml_tag = '!BoxConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        output_str, operator_str, bound_str = node.value.split()  # allows any number of whitespace as separator
        if output_str.find('out') != 0:
            raise ValueError("Malformed BoxConstraint: output descriptor does not start with 'out': " + output_str)
        return BoxConstraint(output_str[3:], operator_str, bound_str)


@yaml_object(property_yaml_context_manager)
class OutputsComparisonConstraint(SingleVarOutputConstraint):
    def __init__(self, output_index_i: int, operator: str, output_index_j: int):
        """
        Create an output constraint in the form

        .. math::
            out_i > out_j
        or

        .. math::
            out_i < out_j
        or

        .. math::
            out_i >= out_j
        or

        .. math::
            out_i <= out_j

        :param output_index_i: The index of the first target output feature of a network (i).
        :param operator: Supported values: less than: '<', 'lt';
         greater than: '>', 'gt'; less than or equals: '<=', '=<', 'le';  greater than or equals: '>=', '=>', 'ge'.
        :param output_index_j: The index of the second target output feature of a network (j).
        """
        if operator != '<' and operator != '>' and operator != "lt" and operator != "gt" \
                and operator != '<=' and operator != '=<' and operator != 'le' \
                and operator != '>=' and operator != '=>' and operator != 'ge':
            raise ValueError("Unknown operator: " + operator + ". Allowed: <, >, lt, gt, <=, =<, le, >=, =>, ge.")
        if output_index_i == output_index_j:
            raise ValueError("Invalid output indices: output_index_i is equal to output_index_j.")
        self.output_index_i = int(output_index_i)
        self.output_index_j = int(output_index_j)
        self.less_than = operator == "<" or operator == "lt" or operator == "<=" or operator == "=<" or operator == "le"
        self.exclude_equals = operator == "<" or operator == ">" or operator == "lt" or operator == "gt"

    def satisfaction_function(self, network_outputs: torch.Tensor, network_inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        value: torch.Tensor
        if self.less_than:  # also <=
            value = network_outputs[:, self.output_index_j] - network_outputs[:, self.output_index_i]
        else:  # greater than (or equals)
            value = network_outputs[:, self.output_index_i] - network_outputs[:, self.output_index_j]
        satisfied: torch.Tensor
        if self.exclude_equals:  # < and >
            satisfied = torch.gt(value, 0)
        else:  # >= and =<
            satisfied = torch.ge(value, 0)
        return value, satisfied

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, OutputsComparisonConstraint):
            return False
        return self.output_index_i == o.output_index_i and self.output_index_j == o.output_index_j \
               and self.less_than == o.less_than \
               and self.exclude_equals == o.exclude_equals

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.output_index_i, self.output_index_j, self.less_than, self.exclude_equals))

    def __repr__(self):
        operator_str: str
        if self.less_than and self.exclude_equals:
            operator_str = '<'
        elif self.less_than and not self.exclude_equals:
            operator_str = '=<'
        elif not self.less_than and self.exclude_equals:
            operator_str = '>'
        else:
            operator_str = '>='
        return f'out{self.output_index_i} {operator_str} out{self.output_index_j}'

    yaml_tag = '!OutputsComparisonConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        output_str_i, operator_str, output_str_j = node.value.split()  # allows any number of whitespace as separator
        if output_str_i.find('out') != 0:
            raise ValueError("Malformed OutputsComparisonConstraint: "
                             "first output descriptor does not start with 'out': " + output_str_i)
        if output_str_j.find('out') != 0:
            raise ValueError("Malformed OutputsComparisonConstraint: "
                             "second output descriptor does not start with 'out': " + output_str_j)
        return BoxConstraint(output_str_i[3:], operator_str, output_str_j[3:])


@yaml_object(property_yaml_context_manager)
class ExtremumConstraint(SingleVarOutputConstraint):
    def __init__(self, output_index: int, operator: str, extremum: str):
        """
        Create an output constraint in the form

        .. math::
            out_i = strict\\_max(out)
        or

        .. math::
            out_{i} \\neq min(out)

        Where strict_max and strict_min mean strict maximum and strict minimum,
        disallowing equal values in the respective set.
        More formally :math:`out_i = strict\\_max(out)` actually means :math:`out_i > \\max_{j\\neq i}(out_j)`
        and :math:`out_i = strict\\_min(out)` means :math:`out_i < \\min_{j\\neq i}(out_j)`.

        :param output_index: The index of the target output feature of a network (i).
        :param operator: Supported values: equals: '==', '=' 'eq';
         not equals: '!=', '~=', '/=', 'ne'.
        :param extremum: Supported values if operator is equals:
         minimum: 'smin', 'strict_min', 'strict_minimum'; 'maximum: 'smax', 'strict_max', 'strict_maximum'.
         If operator is not equals: minimum: 'min', 'minimum'; maximum: 'max', 'maximum'
        """
        self.output_index = int(output_index)

        if operator != '==' and operator != '=' and operator != "eq" \
                and operator != "!=" and operator != "~=" and operator != "/=" and operator != 'ne':
            raise ValueError("Unknown operator: " + operator + ". Allowed: ==, =, eq, !=, ~=, /=, ne")
        self.equals = operator == '==' or operator == '=' or operator == 'eq'

        if self.equals:
            if extremum != 'smin' and extremum != 'smax' and extremum != 'strict_min' and extremum != 'strict_max' \
                    and extremum != 'strict_minimum' and extremum != 'strict_maximum':
                raise ValueError("Unknown extremum: " + extremum + ". Allowed for ==: smin, strict_min, strict_minimum,"
                                                                   "smax, strict_max, strict_maximum")
            self.maximum = extremum == 'smax' or extremum == 'strict_max' or extremum == 'strict_maximum'
        else:
            if extremum != 'min' and extremum != 'max' and extremum != 'minimum' and extremum != 'maximum':
                raise ValueError("Unknown extremum: " + extremum + ". Allowed for !=: min, minimum, max, maximum")
            self.maximum = extremum == 'max' or extremum == 'maximum'

    def satisfaction_function(self, network_outputs: torch.Tensor, network_inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        target_output = network_outputs[:, self.output_index]
        other_outputs = network_outputs[:, torch.arange(network_outputs.shape[1]) != self.output_index]

        value: torch.Tensor
        if self.maximum and self.equals:  # out_i == strict_max(out)
            value = target_output - torch.max(other_outputs, dim=1).values
        elif self.maximum and not self.equals:  # out_i != max(out)
            value = torch.max(other_outputs, dim=1).values - target_output
        elif not self.maximum and self.equals:  # out_i == strict_min(out)
            value = torch.min(other_outputs, dim=1).values - target_output
        else:  # out_i != min(out)
            value = target_output - torch.min(other_outputs, dim=1).values
        return value, torch.gt(value, 0)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ExtremumConstraint):
            return False
        return self.output_index == o.output_index and self.equals == o.equals and self.maximum == o.maximum

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.output_index, self.equals, self.maximum))

    def __repr__(self):
        if self.equals:
            operator_str = '=='
            extremum_str = 'strict_max' if self.maximum else 'strict_min'
        else:
            operator_str = '!='
            extremum_str = 'max' if self.maximum else 'min'
        return f'out{self.output_index} {operator_str} {extremum_str}(out)'

    yaml_tag = '!ExtremumConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        output_str, operator_str, extremum_str = node.value.split()
        if output_str.find('out') != 0:
            raise ValueError("Malformed ExtremumConstraint: output descriptor does not start with 'out': "
                             + output_str)
        if not extremum_str.endswith('(out)'):
            raise ValueError("Malformed ExtremumConstraint: extremum descriptor does not end with '(out)': "
                             + extremum_str)
        return ExtremumConstraint(output_str[3:], operator_str, extremum_str[:-5])


@yaml_object(property_yaml_context_manager)
class MultiOutputExtremumConstraint(SingleVarOutputConstraint):
    def __init__(self, extremum: str, operator: str, output_indices: Sequence[int]):
        """
        Create an output constraint in the form

        .. math::
            strict\\_max(out) \\in \\{ out_i, out_j \\}
        or

        .. math::
            \\min(out) \\notin \\{ out_i, out_j, out_k \\}

        Where strict_max and strict_min are used similarly as for ExtremumConstraint.
        There is however a slight quirk in the their semantics as implemented in this class.
        :math:`strict\\_max(out) \\in \\{ out_i, out_j \\}` is true also if :math:`out_i`
        is maximal and :math:`out_i` equals :math:`out_j`
        (so :math:`out_j` is also maximal). In this case a strict maximum actually does not exist.
        So for this constraint type strict_max actually means (for two target outputs i and j):
        :math:`out_i > \\max_{k \\neq i, k \\neq j}(out_k) \\vee out_j > \\max_{k \\neq i, k \\neq j}(out_k)`
        and equivalently for strict_min.
        For strict extrema and not_in the rule is that the property is not satisfied,
        if the non-strict extremum is contained in both the target output and the remaining ones.
        For not_in strict extrema are not permitted.

        Regarding YAML parsing, this class supports a few options for specifying the target output indices.
        You may use {}, [] and () as index set delimiters or use no delimiters at all.
        Elements of the set need to be separated by commas. The following examples are all valid YAML
        descriptors for a MultiOutputExtremumConstraint:

            strict_max(out) in {out0,out3,out7}

            min(out) not_in [out1, out4, out9 ]

            minimum(out) not_in (out15, out17)

            smax(out) in out0, out1, out12

        :param extremum: Supported values are: 'smin', 'strict_min', 'strict_minimum', 'smax', 'strict_max',
         'strict_maximum', 'min', 'minimum', 'maximum: 'max', 'maximum'
        :param operator: Supported values: 'in', 'not_in'.
        :param output_indices: The indices of the targeted output of the network (i,j and k in the examples).
        """
        # it is required for index access to have a list
        self.output_indices = [int(out_index) for out_index in output_indices]

        if operator != 'in' and operator != 'not_in':
            raise ValueError("Unknown operator: " + operator + ". Allowed: in, not_in")
        self.contained_in = operator == 'in'

        if extremum != 'smin' and extremum != 'strict_min' and extremum != 'strict_minimum' \
                and extremum != 'smax' and extremum != 'strict_max' and extremum != 'struct_maximum' \
                and extremum != 'min' and extremum != 'minimum' and extremum != 'max' and extremum != 'maximum':
            raise ValueError("Unknown extremum: " + extremum + ". Allowed: smin, strict_min, strict_minimum, "
                                                               "smax, strict_max, strict_maximum, min, minimum, "
                                                               "max, maximum. ")
        self.strict = extremum == 'smin' or extremum == 'strict_min' or extremum == 'strict_minimum' \
                      or extremum == 'smax' or extremum == 'strict_max' or extremum == 'strict_maximum'
        if not self.contained_in and self.strict:
            raise ValueError("Invalid configuration strict extremum not in: " + extremum + " " + operator)
        self.maximum = extremum == 'smax' or extremum == 'strict_max' or extremum == 'strict_maximum' \
                       or extremum == 'max' or extremum == 'maximum'

    def satisfaction_function(self, network_outputs: torch.Tensor, network_inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        target_outputs = network_outputs[:, self.output_indices]
        other_outputs = network_outputs[:, [i not in self.output_indices for i in range(network_outputs.shape[1])]]
        value: torch.Tensor
        if self.contained_in and self.maximum:  # max(out) in {out_i, ..., out_j} and strict_max(out) in ...
            value = torch.max(target_outputs, dim=1).values - torch.max(other_outputs, dim=1).values
        elif self.contained_in and not self.maximum:  # min(out) in {out_i, ..., out_j} and strict_min(out) in ...
            value = torch.min(other_outputs, dim=1).values - torch.min(target_outputs, dim=1).values
        elif not self.contained_in and self.maximum:  # max(out) not_in {out_i, ..., out_j} and strict_max not_in ...
            value = torch.max(other_outputs, dim=1).values - torch.max(target_outputs, dim=1).values
        else:  # min(out) not_in {out_i, ..., out_j} and strict_min not_in ...
            value = torch.min(target_outputs, dim=1).values - torch.min(other_outputs, dim=1).values

        if self.strict or not self.contained_in:
            return value, torch.gt(value, 0)
        else:
            return value, torch.ge(value, 0)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MultiOutputExtremumConstraint):
            return False
        return self.output_indices == o.output_indices and self.contained_in == o.contained_in \
               and self.strict == o.strict and self.maximum == o.maximum

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((tuple(self.output_indices), self.contained_in, self.strict, self.maximum))

    def __repr__(self):
        operator_str = 'in' if self.contained_in else 'not_in'
        extremum_str: str
        if self.strict and self.maximum:
            extremum_str = 'strict_max'
        elif self.strict and not self.maximum:
            extremum_str = 'strict_min'
        elif not self.strict and self.maximum:
            extremum_str = 'max'
        else:
            extremum_str = 'min'
        desc = f'{extremum_str}(out) {operator_str} '
        desc += '{'
        for i in self.output_indices:
            desc += f'out{i}'
            desc += ', '
        desc = desc[:-2]
        desc += '}'
        return desc

    yaml_tag = '!MultiOutputExtremumConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        parts = node.value.split()
        extremum_str = parts[0]
        operator_str = parts[1]
        # also split on comma in output strings
        output_strs = list(itertools.chain(*[s.split(',') for s in parts[2:]]))
        # remove all brackets
        output_strs = [
            s.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            for s in output_strs
        ]
        # drop all remaining empty strings
        output_strs = [s for s in output_strs if len(s) > 0]
        # make sure that all start with out and remove those
        for i in range(len(output_strs)):
            if not output_strs[i].startswith('out'):
                raise ValueError("Malformed MultiOutputExtremumConstraint: output descriptor does not start with 'out':"
                                 + extremum_str)
            output_strs[i] = output_strs[i][3:]

        if not extremum_str.endswith('(out)'):
            raise ValueError("Malformed MultiOutputExtremumConstraint: extremum descriptor does not end with '(out)': "
                             + extremum_str)

        return MultiOutputExtremumConstraint(
            extremum_str[:-5], operator_str,
            [int(output_str) for output_str in output_strs]
        )


@yaml_object(property_yaml_context_manager)
class SameExtremumConstraint(MultiVarOutputConstraint):
    """
    A MultiVarOutputConstraint over two variables.

    The constraint expresses that the maximal/minimal output
    for both variables is the same.
    """

    def __init__(self, extremum: str):
        """
        Create a two-variable output constraint in the form

        .. math::
            arg\\_max(out) == arg\\_max(out')
        or similar expressions with min.

        :param extremum: Supported values:
         minimum: 'minimum', 'min'; maximum: 'max', 'maximum'.
        """
        if extremum != 'min' and extremum != 'max' and extremum != 'minimum' and extremum != 'maximum':
            raise ValueError("Unknown extremum: " + extremum + ". Allowed: min, minimum, max, maximum")
        self.maximum = extremum == 'max' or extremum == 'maximum'

    def satisfaction_function(self, network_outputs: Sequence[torch.Tensor], network_inputs: Sequence[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(network_outputs) == 2, "SameExtremumConstraint currently supports only two variables. " \
                                          "Use a conjunction of SameExtremumConstraints for more variables."
        out_1, out_2 = network_outputs
        assert out_1.shape == out_2.shape, "Both network outputs need to have the same shape."
        num_outputs = out_1.shape[1]
        # the expression arg_max(out) == arg_max(out') is equivalent to:
        # OR_{i in 1 ... num_outputs)
        #       AND_{j in 1 ... num_outputs, j!=i} y_i > y_j
        #   AND
        #       AND_{j in 1 ... num_outputs, j!=i} y_i' > y_j'
        # we use the usual OR => max, AND => min construction
        # and take a shortcut for the inner AND
        values = []
        for i in range(num_outputs):
            select_others = torch.arange(num_outputs) != i
            target_output_1 = out_1[:, i]
            target_output_2 = out_2[:, i]
            other_outputs_1 = out_1[:, select_others]
            other_outputs_2 = out_2[:, select_others]
            if self.maximum:  # is out_i maximal?
                value_1 = target_output_1 - torch.max(other_outputs_1, dim=1).values
                value_2 = target_output_2 - torch.max(other_outputs_2, dim=1).values
            else:  # is out_i minimal?
                value_1 = torch.min(other_outputs_1, dim=1).values - target_output_1
                value_2 = torch.min(other_outputs_2, dim=1).values - target_output_2
            value = torch.min(value_1, value_2)
            values.append(value)
        # values are vectors (one entry per batch). vstack adds a row dimension
        # max then removes that row dimension again
        values = torch.max(torch.vstack(values), dim=0).values
        return values, torch.ge(values, 0)  # max and min, not strict max and min

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ExtremumConstraint):
            return False
        return self.maximum == o.maximum

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.maximum)

    def __repr__(self):
        extremum_str = 'max' if self.maximum else 'min'
        return f"{extremum_str}(out) == {extremum_str}(out')"

    yaml_tag = '!SameExtremumConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        string = node.value
        if string == "max(out) == max(out')":
            return SameExtremumConstraint("max")
        elif string == "min(out) == min(out')":
            return SameExtremumConstraint("min")
        else:
            raise ValueError("Malformed SameExtremumConstraint.")


@yaml_object(property_yaml_context_manager)
class OutputDistanceConstraint(MultiVarOutputConstraint):
    """
    A MultiVarOutputConstraint over two variables.

    The constraint expresses that the distance between
    the outputs for the two variables needs to be bounded.

    .. math::
        ||out - out'|| \\leq \\delta
    """

    def __init__(self, threshold: float, norm_order: Union[int, float, str] = 'inf'):
        """
        Create a two-variable output constraint of the form:

        .. math::
            ||out - out'|| \\leq \\delta

        :param threshold: The threshold on the distance of the outputs (delta).
        :param norm_order: The order of the vector norm to use. Any L-norm is supported.
          `norm_order=1` yields the L-1 norm, `norm_order=2` yields the L-2 norm and
          `norm_order="inf"` yields the L-infinity norm.
        """
        assert threshold > 0, f"Threshold needs to be positive. Got: {threshold}"
        self.threshold = threshold
        self.ord = float(norm_order)

    def satisfaction_function(self, network_outputs: Sequence[torch.Tensor], network_inputs: Sequence[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(network_outputs) == 2, "OutputDistanceConstraint requires exactly two variables."
        out_1, out_2 = network_outputs
        assert out_1.shape == out_2.shape, "Both network outputs need to have the same shape."

        non_batch_dims = tuple(range(len(out_1.shape)))[1:]
        distance = torch.linalg.vector_norm(
            out_1 - out_2,
            ord=self.ord,
            dim=non_batch_dims
        )
        return self.threshold - distance, distance <= self.threshold

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, OutputDistanceConstraint):
            return False
        return self.threshold == o.threshold and self.ord == o.ord

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.threshold, self.ord))

    def __repr__(self):
        return f'L_{self.ord} distance <= {self.threshold}'

    yaml_tag = '!OutputDistanceConstraint'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        ord_str, distance_str, comp_str, threshold_str = node.value.split()
        if ord_str[:2] != "L_":
            raise ValueError("Malformed OutputDistanceConstraint: does not start with L_.")
        if distance_str != "distance":
            raise ValueError("Malformed OutputDistanceConstraint.")
        if comp_str != "<=":
            raise ValueError("Malformed OutputDistanceConstraint.")
        return OutputDistanceConstraint(threshold=float(threshold_str), norm_order=ord_str[2:])


# ======================================================================================================================
# ======================================================================================================================
#                                               Input Constraints
# ======================================================================================================================
# ======================================================================================================================


class InputConstraint(ABC):
    """
    A constraint on the input of a neural network.

    Input constraints are (canonically) satisfied whenever the
    satisfaction function is non-negative (>= 0).
    This is different from OutputConstraints, which may require
    strict positivity.

    Implementing `satisfaction_function` is mandatory.
    Implementing the other methods is optional.
    These methods (like `compute_projection`) enable the application
    of certain falsifiers or verifiers or make the application faster.
    """

    @abstractmethod
    def satisfaction_function(self, inputs: Any) -> torch.Tensor:
        """
        Compute whether each element of an input batch satisfies
        the input constraint.

        The input is presented as a tensor where the first dimension
        is the batch dimension (for the purpose of vectorisation).

        Return a 1d tensor (vector) containing the satisfaction
        function value for each batch element at the index
        of the batch element in the batch.

        :param inputs: An input batch.
        :return: A vector with one satisfaction value for each input batch element.
          Negative values correspond to inputs that violate the input constraint.
        """
        raise NotImplementedError()

    def compute_projection(self, inputs: Any) -> torch.Tensor:
        """
        Project inputs to the set of inputs that satisfy the input constraint (feasible set).

        Projection means computing the point in the feasible set that is closest to the
        given input that does not satisfy the constraint.

        The input to this method is batched, like for `satisfaction_function`.

        :param inputs: An input batch to project to the feasible set.
        :return: The projected inputs.
          The return value has the same shape as `inputs`.
        """
        raise NotImplementedError()


class SingleVarInputConstraint(InputConstraint):
    """
    An input constraint that operates on a single input variable.
    """

    @abstractmethod
    def satisfaction_function(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute_projection(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class MultiVarInputConstraint(InputConstraint):
    """
    An input constraint that operates on multiple input variables.
    """

    @abstractmethod
    def satisfaction_function(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def compute_projection(self, inputs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        raise NotImplementedError()


@yaml_object(property_yaml_context_manager)
class DistanceConstraint(MultiVarInputConstraint):
    """
    An input constraint for two inputs that enforces that the two inputs have a distance
    of at most :math:`\\epsilon` (:math:`\\epsilon > 0`).

    Currently, this class only supports the :math:`L_\\infty` distance.

    .. math::
        (x,y) \\in C \\Leftrightarrow ||x - y||_\\infty \\leq \epsilon
    """
    def __init__(self, threshold: float, norm_order: Union[int, float, str] = 'inf'):
        """
        Creates a new DistanceConstraint

        :param threshold: The bound on the distance of two input vectors.
        :param norm_order: The order of the norm corresponding to the distance.
          Currently, only the :math:`L_\\infty` distance/norm is supported.
          Therefore, `norm_order` needs to be `"inf"`.
        """
        assert threshold > 0, f"Threshold needs to be positive. Got: {threshold}"
        norm_order = float(norm_order)
        if norm_order != float('inf'):
            raise ValueError(f"Unsupported norm order: {norm_order}. Currently supported: inf")
        self.ord = float(norm_order)
        self.threshold = threshold

    def _check_inputs(self, inputs: Sequence[torch.Tensor]):
        assert len(inputs) == 2, "DistanceConstraint requires exactly two inputs."
        assert inputs[0].shape == inputs[1].shape, "Both inputs need to have the same shape."

    def satisfaction_function(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        self._check_inputs(inputs)
        # this is just 1, 2, ..., n (the input having n dimensions)
        # the index of every dimension, except the first (index 0)
        non_batch_dims = tuple(range(len(inputs[0].shape)))[1:]
        distance = torch.linalg.vector_norm(
            inputs[0] - inputs[1],
            ord=self.ord,
            dim=non_batch_dims
        )
        return self.threshold - distance

    def compute_projection(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the inputs to satisfy the DistanceConstraint.

        The projection of two vectors to have at most a certain
        :math:`L_\\infty` distance is not unique.
        This method clips the second input to the :math:`L_\\infty` ball
        around the first input.

        :param inputs: The two inputs.
        :return: The first input and the clipped second input.
        """
        self._check_inputs(inputs)
        first: torch.Tensor = inputs[0]
        second: torch.Tensor = inputs[1]
        second = second.clamp(
            min=first - self.threshold,
            max=first + self.threshold
        )
        return first, second

    def __eq__(self, o: object):
        if not isinstance(o, DistanceConstraint):
            return False
        return self.ord == o.ord and self.threshold == o.threshold

    def __ne__(self, o: object):
        return not self.__eq__(o)

    def __hash__(self):
        return hash((self.threshold, self.ord))

    def __repr__(self):
        return f'L_{self.ord} distance <= {self.threshold}'

    yaml_tag = "!DistanceConstraint"

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, data.__repr__())

    @classmethod
    def from_yaml(cls, _, node):
        ord_str, distance_str, comp_str, threshold_str = node.value.split()
        if ord_str[:2] != "L_":
            raise ValueError("Malformed DistanceConstraint: does not start with L_.")
        if distance_str != "distance":
            raise ValueError("Malformed DistanceConstraint.")
        if comp_str != "<=":
            raise ValueError("Malformed DistanceConstraint.")
        return DistanceConstraint(threshold=float(threshold_str), norm_order=ord_str[2:])


# ======================================================================================================================
# ======================================================================================================================
#                                                       Property
# ======================================================================================================================
# ======================================================================================================================

@yaml_object(property_yaml_context_manager)
class Property:
    """
    A property that can be falsified by DeepOpt.
    It consists of inputs bounds and a violation function which encodes some output condition.

    The core of this class is the satisfaction_function method. The input bounds are specified via the constructor.
    """

    def __init__(self,
                 lower_bounds: Dict[int, float],
                 upper_bounds: Dict[int, float],
                 output_constraint: OutputConstraint,
                 input_constraint: Optional[InputConstraint] = None,
                 property_name: str = ""):
        """
        Creates a Property object from the following parameters:

        :param lower_bounds: Lower bounds on the inputs.
         Using a dictionary allows leaving bounds for some inputs unspecified,
         which is useful if there are many irrelevant inputs.
         The keys of the dictionary are the indices of the network inputs (zero indexed).
         The values are the respective lower bounds.

         For example ``{0: -4, 3: 1, 7: 0}`` specifies that the first input (index 0) has a lower bound of -4
         for this property, the fourth input (index 3) has a lower bound of 1 and the 8th input (index 7) has
         a lower bound of 0.
         All other inputs do not have a lower bound.
        :param upper_bounds: Like lower_bounds, but for the upper bounds of the inputs.
        :param output_constraint: The constraint on the network output.
        :param input_constraint: An additional constraint on the set of valid inputs to this property.
          This complements the box-shaped input region.
          Defaults to None, i.e. no further input constraint.
        :param property_name: The name of this property.
         Defaults to the empty string.
        """
        # copy the input bounds and make tuples everywhere
        # self.bounds = tuple((l,u) for l,u in bounds)
        # copy the bounds, so the dicts can not be changed from the outside
        self.lower_bounds = dict(lower_bounds)
        self.upper_bounds = dict(upper_bounds)
        # computed later when network input size is known
        # used to speed up valid_input and input_constraint_projection
        self._lower_bounds_vector = None
        self._upper_bounds_vector = None

        self._input_constraint = input_constraint
        self._output_constraint = output_constraint
        self.property_name = property_name

    @property
    def input_constraint(self):
        # input_constraint was added later on
        # check whether the attribute exists in order to make legacy
        # properties stored in dill files still work.
        if hasattr(self, '_input_constraint'):
            return self._input_constraint
        else:
            return None

    @property
    def output_constraint(self):
        # legacy properties used to have a output_constraint attribute, not a property
        # to still support such (stored in dill files e.g.) do the below
        if hasattr(self, '_output_constraint'):
            return self._output_constraint
        else:
            return self.__dict__['output_constraint']

    def input_bounds(self, network: NeuralNetwork) -> Tuple[Tuple[float, float], ...]:
        """
        Get the input bounds of the property, intersected with the input bounds of a neural network.

        :param network: The neural network with which the input bounds should be intersected.
        :return: The input bounds for the given network and this property as a tuple of an lower and an upper bound
         for each input dimension.
        """
        # first transfer the bounds dictionaries to sequences with -inf and inf where not specified
        lower = []
        upper = []
        for i in range(network.num_inputs()):
            if i in self.lower_bounds:
                lower.append(self.lower_bounds[i])
            else:
                lower.append(-np.inf)
            if i in self.upper_bounds:
                upper.append(self.upper_bounds[i])
            else:
                upper.append(np.inf)

        # make 4-tuples containing network min, network max, property lower bounds, property upper bound
        # for each input dimension
        zipped = zip(network.mins.flatten(), network.maxes.flatten(), lower, upper)
        return tuple(
            (
                max(lower_net.item(), float(lower_prop)),
                min(upper_net.item(), float(upper_prop))
            )
            for lower_net, upper_net, lower_prop, upper_prop in zipped
        )

    def _get_bounds_vectors(self, num_inputs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lb_vector = self._lower_bounds_vector
        ub_vector = self._upper_bounds_vector
        if lb_vector is None or len(lb_vector) != num_inputs:
            # either tensor has not been computed, or has wrong number of bounds
            lb_vector = torch.tensor([
                self.lower_bounds[i] if i in self.lower_bounds else float('-inf')
                for i in range(num_inputs)
            ])
            self._lower_bounds_vector = lb_vector
        if ub_vector is None or len(ub_vector) != num_inputs:
            ub_vector = torch.tensor([
                self.upper_bounds[i] if i in self.upper_bounds else float('inf')
                for i in range(num_inputs)
            ])
            self._upper_bounds_vector = ub_vector
        return lb_vector, ub_vector

    def _input_within_input_bounds(self, network_inputs: torch.Tensor) -> torch.Tensor:
        is_valid = torch.empty(len(network_inputs), dtype=torch.bool)
        is_valid[:] = True
        lb_vector, ub_vector = self._get_bounds_vectors(network_inputs.shape[1])
        is_valid &= torch.all(network_inputs >= lb_vector, dim=1)
        is_valid &= torch.all(network_inputs <= ub_vector, dim=1)
        return is_valid

    def valid_input(self, network_inputs: torch.Tensor) -> torch.Tensor:
        """
        Checks for a batch of network inputs, whether they satisfy the
        input bounds and the input constraint, if present.

        :param network_inputs: The batch of network input tensors to check.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :return: A 1d tensor (vector) indicating whether each of the input batch elements
         lies within the input bounds and satisfies the additional input constraint (when present).
        """
        is_valid = self._input_within_input_bounds(network_inputs)
        if self._input_constraint is not None:
            is_valid &= self._input_constraint.satisfaction_function(network_inputs) >= 0
        return is_valid

    def input_constraint_satisfaction_function(self, network_inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the input constraint satisfaction function for a batch of network inputs.
        The result is non-negative if the input constraint is satisfied.

        :param network_inputs: The batch of network input tensors
         for which to compute the input constraint satisfaction.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :return: A 1d tensor (vector) containing the satisfaction function of the input constraint
         for each input batch element.
        """
        return self.input_constraint.satisfaction_function(network_inputs)

    def _input_bounds_projection(self, network_inputs: torch.Tensor) -> torch.Tensor:
        lb_vector, ub_vector = self._get_bounds_vectors(network_inputs.shape[1])
        return torch.clamp(network_inputs, min=lb_vector, max=ub_vector)

    def input_constraint_projection(self, network_inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the input constraint projection for a batch of network inputs.
        The resulting projected input tensors satisfy the input constraint
        and lie within the input bounds.

        :param network_inputs: The batch of network input tensors to project to the set of valid inputs.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :return: The closest inputs to the given inputs that are valid for this property.
        """
        projected = network_inputs
        if self.input_constraint is not None:
            projected = self.input_constraint.compute_projection(network_inputs)
        return self._input_bounds_projection(projected)

    def satisfaction_function(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        """
        Returns a tensor measuring the satisfaction or violation of this property for the network outputs
        calculated using the given batch of inputs.
        Minimize this function (for each vector element individually) to obtain violations of this property.

        The exact semantics whether this property is violated are encoded in the ``property_satisfied`` method.
        You should always use this method to check if a minimum of this method corresponds to a
        property violation.

        Implementations need to support automatic gradient calculation with pytorch and processing batches of
        network_outputs.
        All ``network_inputs`` are presented as batches of network inputs.
        Also for a single set of inputs, ``network_inputs`` will have at least two dimensions, with the first
        dimension being the batch dimension.
        In this case, that dimension will contain only a single element.

        Satisfaction functions need to have the boundary between satisfaction and violation at 0.
        The value of 0 can indicate either satisfaction or violation.
        This information is supplied via the ``property_satisfied`` function.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network: A network to evaluate.
         The inputs will be fed to the network once or multiple times.
        :return: A 1-d tensor (vector) measuring the satisfaction of this property for each element of the
         ``network_outputs`` batch individually.
         Smaller values correspond to
         more closeness to the property being violated.
        """
        network_outputs = self.calc_network_outputs(network_inputs, network)
        return self.satisfaction_from_internal(network_inputs, network_outputs)

    def property_satisfied(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        """
        Returns whether this property is satisfied for the given batch of network outputs,
        assuming that the inputs to the network were in this properties input_bounds.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network: A network to evaluate.
         The inputs will be fed to the network once or multiple times.
        :return: A vector of booleans stating whether this property is satisfied
         for each of the network outputs in the given batch.
        """
        network_outputs = self.calc_network_outputs(network_inputs, network)
        return self.property_satisfied_from_internal(network_inputs, network_outputs)

    def full_witness(self, network_inputs: torch.Tensor, network: NeuralNetwork) \
            -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        """
        Compute internal network outputs, satisfaction and whether the property is satisfied.
        Returns network inputs, network outputs, satisfaction function output and whether
        the property is satisfied (in this order)

        Have a look at `satisfaction_function` for more information on how to use this
        function.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network: A network to evaluate.
         The inputs will be fed to the network once or multiple times.
        :return: Tuple of network inputs, internal network outputs, satisfaction function value and
         property satisfied truth value.
        """
        network_outputs = self.calc_network_outputs(network_inputs, network)
        sat_value, is_sat = self.output_constraint.satisfaction_function(network_outputs, network_inputs)
        return network_inputs, network_outputs, sat_value, is_sat

    def calc_network_outputs(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> Any:
        """
        Calculates one or multiple network outputs as an internal witness of violation or satisfaction.
        The information can be used with the `satisfaction_from_internal` and `property_satisfied_from_internal`
        methods that accept the calculated construction of network outputs as input.

        The same rules regarding the batch dimension apply for `network_inputs` as in `satisfaction_function`.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network: A network to evaluate.
         The inputs will be fed to the network once or multiple times.
        :return: Some object composed of network outputs.
        """
        return network(network_inputs)

    def calc_network_outputs_tensor(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        """
        Same as `calc_network_outputs`, but returns the result as a tensor.
        The result of `calc_network_outputs` may be any object for internal handling.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network: A network to evaluate.
         The inputs will be fed to the network once or multiple times.
        :return: A tensor composed of network outputs.
        """
        return self.calc_network_outputs(network_inputs, network)

    def satisfaction_from_internal(self, network_inputs: torch.Tensor, network_outputs: Any) -> torch.Tensor:
        """
        Calculates the satisfaction function from network outputs calculated using `calc_network_outputs`.

        The same rules as for `satisfaction_function` regarding the batch dimension and otherwise apply.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network_outputs: Network outputs calculated using `calc_network_outputs` for the `network_inputs`.
        :return: A 1-d tensor containing one satisfaction value per batch entry.
        """
        return self.output_constraint.satisfaction_function(
            network_outputs, network_inputs
        )[0]

    def property_satisfied_from_internal(self, network_inputs: torch.Tensor, network_outputs: Any) \
            -> torch.Tensor:
        """
        Calculates whether the property is satisfied from network outputs calculated using `calc_network_outputs`.

        The same rules as for `property_satisfied` regarding the batch dimension and otherwise apply.

        :param network_inputs: The batch of network input tensors to investigate for violations.
         These inputs need to lie within the input bounds of this property.
         The first dimension is the batch dimension.
         This dimension needs to be present.
        :param network_outputs: Network outputs calculated using `calc_network_outputs` for the `network_inputs`.
        :return: A 1-d tensor of booleans containing whether the property is satisfied for each batch entry.
        """
        return self.output_constraint.satisfaction_function(network_outputs, network_inputs)[1]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Property):
            return False
        return self.lower_bounds == o.lower_bounds and self.upper_bounds == o.upper_bounds \
            and self.input_constraint == o.input_constraint and self.output_constraint == o.output_constraint \
            and self.property_name == o.property_name

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((
            tuple(self.lower_bounds.items()), tuple(self.upper_bounds.items()),
            self.input_constraint, self.output_constraint, self.property_name
        ))

    def __str__(self):
        return self.property_name

    yaml_tag = "!Property"

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = {
            'name': data.property_name,
            'input_lower_bounds': data.lower_bounds,
            'input_upper_bounds': data.upper_bounds,
            'output_constraint': data.output_constraint
        }
        if data.input_constraint is not None:
            mapping['input_constraint'] = data.input_constraint
        return dumper.represent_mapping(Property.yaml_tag, mapping)

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        if 'input_constraint' not in mapping:
            mapping['input_constraint'] = None
        if len(mapping) != 5:
            raise ValueError(f"Malformed Property: wrong number of attributes: {mapping}")
        return Property(mapping['input_lower_bounds'], mapping['input_upper_bounds'],
                        mapping['output_constraint'], mapping['input_constraint'],
                        mapping['name'])


@yaml_object(property_yaml_context_manager)
class MultiVarProperty(Property):
    """
    A property with multiple input variables that can be falsified by DeepOpt.

    The multiple input variables are stacked to form one input.
    """

    def __init__(self,
                 lower_bounds: Sequence[Dict[int, float]],
                 upper_bounds: Sequence[Dict[int, float]],
                 numbers_of_inputs: Optional[Sequence[int]],
                 output_constraint: OutputConstraint,
                 input_constraint: Optional[MultiVarInputConstraint] = None,
                 property_name: str = ""):
        """
        Creates a MultiInputProperty object from the following parameters:

        :param lower_bounds: The input bounds for all input variables.
          The length of the sequence determines the number of input variables.
          This length needs to match the length of upper_bounds.
        :param upper_bounds: Like lower_bounds, but for the upper bounds of the inputs.
          The length of this sequence needs to match the length of lower_bounds.
        :param numbers_of_inputs: The number of inputs of each of the input variables.
          This information is required for stacking.
          You may set this to None if you provide complete lower_bounds and upper_bounds.
          This means that you provide a lower and upper bound for every single input of each input variable.
          In that case, the numbers of inputs will be inferred.
        :param output_constraint: The constraint on the network output.
          Either a MultiVarOutputConstraint or an aggregate of such constraints (Or, And).
        :param input_constraint: An additional constraint on the set of valid inputs to this property.
          This complements the box-shaped input region.
          Defaults to None, i.e. no further input constraint.
          This input constraint needs to be a MultiVarInputConstraint.
        :param property_name: The name of this property.
        """
        assert len(lower_bounds) == len(upper_bounds), "lower_bounds and upper_bounds need to have the same length."
        if numbers_of_inputs is None:
            numbers_of_inputs = tuple(len(bounds) for bounds in lower_bounds)
        else:
            numbers_of_inputs = tuple(numbers_of_inputs)
        # stack all inputs => remember the first indices of all separate inputs in the stacked vector
        start_ids = np.cumsum((0, ) + numbers_of_inputs)
        self.start_ids = tuple(start_ids[:-1])  # the last element is the total length of stacked bounds
        self.end_ids = tuple(start_ids[1:])
        stacked_lower = {}
        stacked_upper = {}
        for lower, upper, start_id in zip(lower_bounds, upper_bounds, self.start_ids):
            for i, val in lower.items():
                stacked_lower[start_id + i] = val
            for i, val in upper.items():
                stacked_upper[start_id + i] = val
        super().__init__(stacked_lower, stacked_upper, output_constraint, input_constraint, property_name)

    def input_bounds(self, network: NeuralNetwork) -> Tuple[Tuple[np.float32, np.float32], ...]:
        """
        Get the input bounds of the property, intersected with the input bounds of a neural network.

        :param network: The neural network with which the input bounds should be intersected.
        :return: The input bounds for the given network and this property as a tuple of a lower and an upper bound
         for each input dimension.
        """

        # we need to do this separately for each of the input variables
        stacked = []
        for start_id in self.start_ids:
            # complement the lower and upper bounds with -inf/inf where unspecified
            lower = []
            upper = []
            for i in range(network.num_inputs()):
                if start_id + i in self.lower_bounds:
                    lower.append(self.lower_bounds[start_id + i])
                else:
                    lower.append(-np.inf)
                if start_id + i in self.upper_bounds:
                    upper.append(self.upper_bounds[start_id + i])
                else:
                    upper.append(np.inf)
            # compare with the network bounds (see in Property)
            zipped = zip(network.mins.flatten(), network.maxes.flatten(), lower, upper)
            bounds = tuple(
                (np.float32(max(lower_net, lower_prop)), np.float32(min(upper_net, upper_prop)))
                for lower_net, upper_net, lower_prop, upper_prop in zipped
            )
            stacked.extend(bounds)
        return tuple(stacked)

    def _split_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Splits a tensor of input or output batches.
        """
        return tuple(tensor[:, start:end] for start, end in zip(self.start_ids, self.end_ids))

    def valid_input(self, network_inputs: torch.Tensor) -> torch.Tensor:
        is_valid = self._input_within_input_bounds(network_inputs)
        if self._input_constraint is not None:
            inputs_split = self._split_tensor(network_inputs)
            is_valid &= self._input_constraint.satisfaction_function(inputs_split) >= 0
        return is_valid

    def input_constraint_satisfaction_function(self, network_inputs: torch.Tensor) -> torch.Tensor:
        inputs_split = self._split_tensor(network_inputs)
        return self.input_constraint.satisfaction_function(inputs_split)

    def input_constraint_projection(self, network_inputs: torch.Tensor) -> torch.Tensor:
        projected = network_inputs
        if self.input_constraint is not None:
            inputs_split = self._split_tensor(network_inputs)
            projected: Sequence[torch.Tensor] = self.input_constraint.compute_projection(inputs_split)
            projected = torch.hstack(tuple(projected))
        return self._input_bounds_projection(projected)

    def satisfaction_function(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        inputs_split = self._split_tensor(network_inputs)
        network_outputs = tuple(network(inputs) for inputs in inputs_split)
        return self.output_constraint.satisfaction_function(network_outputs, inputs_split)[0]

    def property_satisfied(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        inputs_split = self._split_tensor(network_inputs)
        network_outputs = tuple(network(inputs) for inputs in inputs_split)
        return self.output_constraint.satisfaction_function(network_outputs, inputs_split)[1]

    def full_witness(self, network_inputs: torch.Tensor, network: NeuralNetwork) \
            -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        inputs_split = self._split_tensor(network_inputs)
        network_outputs = tuple(network(inputs) for inputs in inputs_split)
        sat_value, is_sat = self.output_constraint.satisfaction_function(network_outputs, inputs_split)
        return network_inputs, network_outputs, sat_value, is_sat

    def calc_network_outputs(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> Any:
        """
        Calculates network outputs for each variable in network_inputs separately and
        returns them as a tuple.
        """
        inputs_split = self._split_tensor(network_inputs)
        network_outputs = tuple(network(inputs) for inputs in inputs_split)
        return network_outputs

    def calc_network_outputs_tensor(self, network_inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
        """
        Calculates network outputs stacked as a tensor in an additional terminal dimension.
        """
        outputs_as_tuple = self.calc_network_outputs(network_inputs, network)
        return torch.stack(outputs_as_tuple, dim=-1)

    def satisfaction_from_internal(self, network_inputs: torch.Tensor, network_outputs: torch.Tensor) -> torch.Tensor:
        inputs_split = self._split_tensor(network_inputs)
        return self.output_constraint.satisfaction_function(network_outputs, inputs_split)[0]

    def property_satisfied_from_internal(self, network_inputs: torch.Tensor, network_outputs: torch.Tensor) -> torch.Tensor:
        inputs_split = self._split_tensor(network_inputs)
        return self.output_constraint.satisfaction_function(network_outputs, inputs_split)[1]

    def __eq__(self, o: object) -> bool:
        # Note start_ids and length of bounds already has all the information of end_ids
        if not isinstance(o, MultiVarProperty):
            return False
        return self.lower_bounds == o.lower_bounds and self.upper_bounds == o.upper_bounds \
            and self.start_ids == o.start_ids and self.output_constraint == o.output_constraint \
            and self.property_name == o.property_name

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((
            tuple(self.lower_bounds.items()), tuple(self.upper_bounds.items()),
            self.start_ids, self.output_constraint, self.property_name
        ))

    def __str__(self):
        return self.property_name

    yaml_tag = "!MultiVarProperty"

    @classmethod
    def to_yaml(cls, dumper, data):
        split_lower_bounds = []
        split_upper_bounds = []
        for start_id, end_id in zip(data.start_ids, data.end_ids):
            lower = {}
            upper = {}
            for i in range(end_id - start_id):
                if start_id + i in data.lower_bounds:
                    lower[i] = data.lower_bounds[start_id + i]
                if start_id + i in data.upper_bounds:
                    upper[i] = data.upper_bounds[start_id + i]
            split_lower_bounds.append(lower)
            split_upper_bounds.append(upper)
        # start and end are int64s from numpy, which YAML can't handle...
        numbers_of_inputs = [int(end - start) for start, end in zip(data.start_ids, data.end_ids)]
        mapping = {
            'name': data.property_name,
            'input_lower_bounds': split_lower_bounds,
            'input_upper_bounds': split_upper_bounds,
            'numbers_of_inputs': numbers_of_inputs,
            'output_constraint': data.output_constraint
        }
        if data.input_constraint is not None:
            mapping['input_constraint'] = data.input_constraint
        return dumper.represent_mapping(MultiVarProperty.yaml_tag, mapping)

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        if 'input_constraint' not in mapping:
            mapping['input_constraint'] = None
        if len(mapping) != 6:
            raise ValueError(f"Malformed Property: wrong number of attributes: {mapping}")

        return MultiVarProperty(mapping['input_lower_bounds'], mapping['input_upper_bounds'],
                                mapping['numbers_of_inputs'], mapping['output_constraint'],
                                mapping['input_constraint'], mapping['name'])


# ======================================================================================================================
# ======================================================================================================================
#                                                      Factories
# ======================================================================================================================
# ======================================================================================================================


@yaml_object(property_yaml_context_manager)
class RobustnessPropertyFactory:
    """
    Robustness properties are properties that state that for a certain region around some input sample
    all inputs need to be classified the same. <br>
    This class allows generating Properties for robustness of a whole data set.<br>
    The regions are defined as all input samples which have at most a certain L_infty distance epsilon
    around an input image. This means that the maximum difference between one input feature of the sample and the
    inputs from the region is smaller than or equal to epsilon. This size is encoded in the input bound of the
    generated properties.
    """

    def __init__(self, name_prefix: Optional[str] = None, eps: Optional[float] = None,
                 desired_extremum: str = 'strict_max'):
        """
        Create a new RobustnessPropertyFactory, either empty or with certain values.

        :param name_prefix: The name for the property. The generated properties will have names of the form
         "given name #X" where X is the index of the sample in the data set passed to get_properties.
         If None is used, the properties will all have an empty name.
        :param eps: The size of the region around an input sample.
        :param desired_extremum: The extremum that the generated properties will require the right output to be.
         The default value 'strict_max' corresponds to requiring that all input samples within the input region
         are classified as the label of the respective sample, assuming that the maximal output corresponds to the label
         that is assigned to the input sample. If your classification mode is using the minimal output as label, then
         set this to 'strict_min'. All values supported for ExtremumConstraints with equals as operator are supported.
        """
        self._name: Optional[str] = name_prefix
        self._eps: Optional[float] = None
        if eps is not None:
            # this method checks a check if the value is permitted
            self.eps(eps)
        self._extremum: str = ''
        # this method also performs a check
        self.desired_extremum(desired_extremum)

    def eps(self, value: float) -> 'RobustnessPropertyFactory':
        """
        Set the size of the region around an input sample for the generated properties.

        :param value: The permitted change in an input sample.
        :return: the same object the method was called on, having the value of epsilon updated.
        """
        if value <= 0:
            raise ValueError(f"eps value needs to be larger than zero, but wasn't: value={value}")
        self._eps = value
        return self

    def name_prefix(self, prefix: Optional[str]) -> 'RobustnessPropertyFactory':
        """
        Set the name prefix of the generated properties. The generated properties will have names of the form
        "given name #X" where X is the index of the sample in the data set passed to get_properties.
        If None is used, the properties will all have an empty name.

        :param prefix: The prefix of the names of the generated properties.
         Use None to give all generated properties and empty name.
        :return: the same object the method was called on, having the value of epsilon updated.
        """
        self._name = prefix
        return self

    def desired_extremum(self, extremum: str) -> 'RobustnessPropertyFactory':
        """
        Define the extremum that the generated properties will require the right output to be.
        The default value 'strict_max' corresponds to requiring that all input samples within the input region
        are classified as the label of the respective sample, assuming that the maximal output corresponds to the label
        that is assigned to the input sample. If your classification mode is using the minimal output as label, then
        set this to 'strict_min'. All values supported for ExtremumConstraints with equals as operator are supported.

        :param extremum: An extremum descriptor as ExtremumConstraint permits for the operator ==.
        :return: the same object the method was called on, having the classification mode updated.
        """
        # check if the extremum is a valid string
        ExtremumConstraint(0, '==', extremum)
        self._extremum = extremum
        return self

    def get_properties(self, input_samples: Union[np.ndarray, torch.tensor], labels: Sequence[int]) \
            -> Iterator[Property]:
        """
        Generates properties for a set of samples, that together constitute the robustness specification.

        For each sample exactly one property is generated.

        :param input_samples: The inputs of the samples for which properties will be generated. These are used
         together with the previously defined epsilon to define the input bounds of the properties.

         The input samples have to be passed as a 2d array where the columns correspond to the features
         of one input sample and the rows correspond to the different samples.
        :param labels: The labels/classes of the samples that will be encoded into the properties.
         The labels need to be specified as the indices of the corresponding outputs of the network.
         The generated properties
         will require that the maximal (or minimal) output for an input region is the output that is given here.
         Whether maximum or minimum will be required depends on the mode that was set using the
         classify_as method or the constructor. <br>
         Usually the labels will be the predictions of the network that is to be checked.
        :return: A list of properties, one for each sample,
         together constituting the robustness specification
         for the data set.
        """
        if self._eps is None:
            raise ValueError('Eps needs to be set before properties can be generated.')
        if len(input_samples.shape) < 2:
            raise ValueError(f'input_samples needs to have at least two dimensions.')
        if input_samples.shape[0] != len(labels):
            raise ValueError(f'Number of samples and labels does not match: #input_samples={input_samples.shape[0]}, '
                             f'#labels={len(labels)}')

        for index, inputs, label in zip(range(len(labels)), input_samples, labels):
            lower_bounds = dict(enumerate(map(float, inputs.flatten() - self._eps)))
            upper_bounds = dict(enumerate(map(float, inputs.flatten() + self._eps)))
            constraint = ExtremumConstraint(label, '==', self._extremum)
            if self._name is None:
                yield Property(lower_bounds, upper_bounds, constraint)
            else:
                yield Property(lower_bounds, upper_bounds, constraint, property_name=f'{self._name} #{index}')

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RobustnessPropertyFactory):
            return False
        return self._name == o._name and self._eps == o._eps and self._extremum == o._extremum

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self._name, self._eps, self._extremum))

    yaml_tag = "!RobustnessPropertyFactory"

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(RobustnessPropertyFactory.yaml_tag, {
            'name_prefix': data._name,
            'eps': data._eps,
            'desired_extremum': data._extremum
        })

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        if len(mapping) != 3:
            raise ValueError(f"Malformed RobustnessPropertyFactory: wrong number of attributes: {mapping}")
        return RobustnessPropertyFactory(mapping['name_prefix'], mapping['eps'], mapping['desired_extremum'])


@yaml_object(property_yaml_context_manager)
class RegressionRobustnessPropertyFactory:
    """
    Regression robustness properties are properties that state that for a
    certain region around some input sample the output needs to be close
    to the output for the input sample.

    This factory uses the L_infinity distance for both the input set and the
    output set.
    """
    def __init__(
        self, name_prefix: Optional[str] = None, eps: Optional[float] = None,
        delta: Optional[float] = None
    ):
        """
        Create a new RegressionRobustnessPropertyFactory, either empty or
        with certain values.

        :param name_prefix: The name for the property. The generated properties will have names of the form
         "given name #X" where X is the index of the sample in the data set passed to get_properties.
         If None is used, the properties will all have an empty name.
        :param eps: The size of the region around an input sample.
        :param delta: The allowable deviation from the output for an input sample.
        """
        self._name: Optional[str] = name_prefix
        self._eps: Optional[float] = None
        self._delta: Optional[float] = None
        if eps is not None:
            # this method checks a check if the value is permitted
            self.eps(eps)
        if delta is not None:
            # also performs a check
            self.delta(delta)

    def eps(self, value: float) -> 'RegressionRobustnessPropertyFactory':
        """
        Set the size of the region around an input sample for the
        generated properties.

        :param value: The permitted change in an input sample.
        :return: the same object the method was called on,
        having the value of epsilon updated.
        """
        if value <= 0:
            raise ValueError(f"eps value needs to be larger than zero, but wasn't: value={value}")
        self._eps = value
        return self

    def delta(self, value: float) -> 'RegressionRobustnessPropertyFactory':
        """
        Set the allowable deviation of the output from the output for an
        input sample for the generated properties.

        :param value: The permitted change of the output.
        :return: the same object the method was called on,
        having the value of delta updated.
        """
        if value <= 0:
            raise ValueError(f"delta value needs to be larger than zero, but wasn't: value={value}")
        self._delta = value
        return self

    def name_prefix(self, prefix: Optional[str]) -> 'RegressionRobustnessPropertyFactory':
        """
        Set the name prefix of the generated properties. The generated properties will have names of the form
        "given name #X" where X is the index of the sample in the data set passed to get_properties.
        If None is used, the properties will all have an empty name.

        :param prefix: The prefix of the names of the generated properties.
         Use None to give all generated properties and empty name.
        :return: the same object the method was called on, having the value of epsilon updated.
        """
        self._name = prefix
        return self

    def get_properties(
        self, input_samples: Union[np.ndarray, torch.tensor], predictions: Sequence[float]
    ) -> Iterator[Property]:
        """
        Generates properties for a set of samples, that together constitute the
        robustness specification.

        Generates exactly one property for each sample.

        :param input_samples: The inputs of the samples for which properties
         will be generated. These are used together with the previously defined
         epsilon to define the input bounds of the properties.

         The input samples have to be passed as a 2d array where the columns correspond to the features
         of one input sample and the rows correspond to the different samples.
        :param predictions: The predictions/outputs for the samples that will
         be encoded into the properties.
        :return: A list of properties, one for each sample,
         together constituting the robustness specification for the data set.
        """
        if self._eps is None:
            raise ValueError('Eps needs to be set before properties can be generated.')
        if self._delta is None:
            raise ValueError('Delta needs to be set before properties can be generated.')
        if len(input_samples.shape) < 2:
            raise ValueError(f'input_samples needs to have at least two dimensions.')
        if input_samples.shape[0] != len(predictions):
            raise ValueError(f'Number of samples and predictions does not match: #input_samples={input_samples.shape[0]}, '
                             f'#labels={len(predictions)}')

        for index, inputs, prediction in zip(range(len(predictions)), input_samples, predictions):
            lower_bounds = dict(enumerate(map(float, inputs.flatten() - self._eps)))
            upper_bounds = dict(enumerate(map(float, inputs.flatten() + self._eps)))
            constraint = ConstraintAnd(
                BoxConstraint(0, "<=", prediction + self._delta),
                BoxConstraint(0, ">=", prediction - self._delta)
            )
            if self._name is None:
                yield Property(lower_bounds, upper_bounds, constraint)
            else:
                yield Property(lower_bounds, upper_bounds, constraint, property_name=f'{self._name} #{index}')

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RegressionRobustnessPropertyFactory):
            return False
        return self._name == o._name and self._eps == o._eps and self._delta == o._delta

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self._name, self._eps, self._delta))

    yaml_tag = "!RegressionRobustnessPropertyFactory"

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(RobustnessPropertyFactory.yaml_tag, {
            'name_prefix': data._name,
            'eps': data._eps,
            'delta': data._delta,
        })

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        if len(mapping) != 3:
            raise ValueError(f"Malformed RegressionRobustnessPropertyFactory: "
                             f"wrong number of attributes: {mapping}")
        return RobustnessPropertyFactory(mapping['name_prefix'], mapping['eps'], mapping['delta'])


# ======================================================================================================================
# ======================================================================================================================
#                                              Specification Serialisation
# ======================================================================================================================
# ======================================================================================================================


def dump_specification(spec: Sequence[Union[Property, RobustnessPropertyFactory]],
                       stream=None, as_multiple_documents=True, **kwargs) \
        -> Union[str, bytes]:
    """
    Convert a specification (e.g. list or tuple of properties) into a yaml serialization.

    This method uses `ruamel.yaml.YAML('safe', pure=True).dump_all` or `dump` in the background.

    :param spec: The specification to serialize
    :param stream: Where to output the serialization. All options supported by `ruamel.yaml's` `dump_all` are supported.
     If None is passed, sys.stdout is used.
    :param as_multiple_documents: Whether to create multiple yaml documents each containing a property
     (default behavior). If set to False, this method will produce a yaml list of properties.<br>
     This method uses `dump_all` if as_multiple_documents is True and `dump` otherwise.
    :param kwargs: Further options to `dump_all` or `dump`
    """
    if stream is None:
        stream = sys.stdout

    if as_multiple_documents:
        return property_yaml_context_manager.dump_all(spec, stream, **kwargs)
    else:
        return property_yaml_context_manager.dump(spec, stream, **kwargs)


def load_specification(stream) -> Tuple[Union[Property, RobustnessPropertyFactory], ...]:
    """
    Extract all Properties from a yaml stream containing one or multiple documents, each containing
    a single property or a list of properties. Documents of lists of properties and single property documents
    may be mixed in the input stream.

    Anything in the stream that is not a Property will be ignored in the output of this function.

    :param stream: The yaml source to read from.
    :return: All properties from the YAML input stream.
    """
    # this might also contain other top level objects (e.g. defining pi as an anchor)
    objects = tuple(property_yaml_context_manager.load_all(stream))
    properties = []
    for obj in objects:
        if isinstance(obj, Property) or isinstance(obj, RobustnessPropertyFactory) or isinstance(obj, RegressionRobustnessPropertyFactory):
            properties.append(obj)
        elif isinstance(obj, list):
            properties += [obj2 for obj2 in obj
                           if isinstance(obj2, Property) or isinstance(obj2, RobustnessPropertyFactory) or isinstance(obj2, RegressionRobustnessPropertyFactory)]
    return tuple(properties)
