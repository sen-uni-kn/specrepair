import time
import traceback
from copy import deepcopy
from logging import warning
from typing import Tuple, Optional, Sequence, List, Union, Dict

import numpy as np
import itertools
from tempfile import TemporaryDirectory
import os

import torch

from deep_opt import NeuralNetwork, Property
from deep_opt.models.property import OutputConstraint, BoxConstraint, OutputsComparisonConstraint, ExtremumConstraint, \
    MultiOutputExtremumConstraint, ConstraintOr, ConstraintAnd
from nn_repair.utils.shift_property import ShiftedOutputConstraint

from verification_interface import verify_acasxu_style, verify_plain, compute_bounds

from nn_repair import CounterexampleGenerator, Counterexample


def property_constraint_to_eran(constraint: OutputConstraint, num_outputs: int,
                                means_outputs: np.ndarray, ranges_outputs: np.ndarray) -> List[List[Tuple]]:
    """
    Converts a property constraint to the ERAN format.
    The ERAN format is a list of lists of tuples representing a conjunctive normal form formula
    over atoms on the outputs. The outer list contains expressions that all should old (their
    and combination should be true). Each inner list contains atoms of which one should hold
    (their or combination should be true). The atoms are expressions like output 1 greater than output 3
    or output 4 smaller than or equal 75.3.

    Box constraints with other operators than less-than-or-equal are not supported.

    For use in the linear programming minimal modification backend this method supports ShiftedOutputConstraints.
    For BoxConstraints the output bound is shifted. For ExtremumConstraints the shift is recorded
    in the unused third tuple element. ERAN will ignore this, other code may use it.

    Extremum constraints with string and non strict maxima are handled in the same way as ERAN does not differentiate
    here.

    :param constraint: The constraint to convert.
    :param num_outputs: Specifies how many outputs exist in the targeted networks.
    :param means_outputs: The networks output normalization means. Needed to transform constants.
    :param ranges_outputs: The networks output normalization ranges. Needed to transform constants.
    :return: An ERAN constraint representation of the given output constraint.
    """
    # prioritize custom (probably optimised) implementations
    if hasattr(constraint, "eran_constraint") and callable(getattr(constraint, "eran_constraint")):
        return constraint.eran_constraint()
    elif isinstance(constraint, ConstraintAnd):
        # merge outer lists
        return list(itertools.chain(
            *(property_constraint_to_eran(constr, num_outputs, means_outputs, ranges_outputs)
              for constr in constraint.constraints)
        ))
    elif isinstance(constraint, ConstraintOr):
        # essentially we need to convert disjunctive normal form to conjunctive normal form here
        # simply apply the distributive law. The result is by no means minimal.
        # Note: it would make sense to find minimal formulas here, runtime is increased,
        #       especially for MILP. For larger or constraints this would probably be prohibitive
        #       but often it might be sufficient to use a MultiOutputExtremumConstraint instead of OR
        #       and that does not increase runtime.
        product = itertools.product(
            *(property_constraint_to_eran(constr, num_outputs, means_outputs, ranges_outputs)
              for constr in constraint.constraints)
        )
        list_format = [list(itertools.chain(*tpl)) for tpl in product]
        return list_format
    elif isinstance(constraint, BoxConstraint):
        # bound needs to be transformed using the normalization values,
        # since ERAN does not take care of output normalization
        normalized_bound = \
            (constraint.bound - means_outputs[constraint.output_index]) / ranges_outputs[constraint.output_index]
        if not constraint.exclude_equals and constraint.less_than:  # <=
            return [[(constraint.output_index, -1, normalized_bound)]]
        elif constraint.exclude_equals and constraint.less_than:  # <
            return [[(constraint.output_index, -2, normalized_bound)]]
        elif not constraint.exclude_equals and not constraint.less_than:  # >=
            return [[(constraint.output_index, -3, normalized_bound)]]
        else:  # >
            return [[(constraint.output_index, -4, normalized_bound)]]
    elif isinstance(constraint, ExtremumConstraint) or isinstance(constraint, MultiOutputExtremumConstraint):
        out_indices: Sequence
        equals_or_in: bool
        # both constraint types have the field maximum
        if isinstance(constraint, ExtremumConstraint):
            out_indices = [constraint.output_index]
            equals_or_in = constraint.equals
        else:
            out_indices = constraint.output_indices
            equals_or_in = constraint.contained_in
            if constraint.contained_in and not constraint.strict:
                raise ValueError("ERAN only handles strict contained-in MultiOutputExtremumConstraints.")

        if equals_or_in and constraint.maximum:  # max in outputs / out = max
            return [[(out_index, other, 0) for out_index in out_indices]
                    for other in range(num_outputs) if other not in out_indices]
        elif equals_or_in and not constraint.maximum:  # min in outputs / out = min
            return [[(other, out_index, 0) for out_index in out_indices]
                    for other in range(num_outputs) if other not in out_indices]
        elif not equals_or_in and constraint.maximum:  # max not in outputs / out != max
            return [[(other, out_index, 0) for other in range(num_outputs) if other not in out_indices]
                    for out_index in out_indices]
        else:  # min not in outputs / out != min
            return [[(out_index, other, 0) for other in range(num_outputs) if other not in out_indices]
                    for out_index in out_indices]
    elif isinstance(constraint, OutputsComparisonConstraint):
        if constraint.less_than:
            return [[(constraint.output_index_j, constraint.output_index_i, 0)]]
        else:
            return [[(constraint.output_index_i, constraint.output_index_j, 0)]]
    elif isinstance(constraint, ShiftedOutputConstraint):
        original_translated = property_constraint_to_eran(constraint.original,
                                                          num_outputs, means_outputs, ranges_outputs)
        return [[(i, j, k + constraint.offset) for i, j, k in and_list] for and_list in original_translated]
    else:
        raise ValueError(f"Unsupported constraint: {constraint}.\n"
                         f"Either use ConstraintAnd, ConstraintOr, BoxConstraint, OutputsComparisonConstraint, "
                         f"ExtremumConstraint, MultiOutputExtremumConstraint, or ShiftedOutputConstraint, or"
                         f"supply a 'eran_constraint()' method that can be used to verify this constraint with ERAN.")


def eran_comparison_to_property(eran_constraint: Tuple[int, int, float],
                                original_property: Property) -> Property:
    """
    Generates a Property from an eran form atomic comparison.
    This is only concerned with a comparison of two outputs or an output and a constant.

    :param eran_constraint: The eran form output comparison that needs to be converted to an OutputConstraint.
    :param original_property: The original Property. Used for bounds and name of the returned property.
    :return:
    """
    i, j, k = eran_constraint
    if j < 0:
        # this constraint is comparing an output with a constant
        # operators:
        # -1: <=
        # -2: <
        # -3: >=
        # -4: >
        if j == -1:
            operator = '<='
        elif j == -2:
            operator = '<'
        elif j == -3:
            operator = '>='
        else:
            operator = '>'
        output_constraint = BoxConstraint(i, operator, k)
    else:
        # yi > yj
        output_constraint = OutputsComparisonConstraint(i, '>', j)
    return Property(
        lower_bounds=original_property.lower_bounds,
        upper_bounds=original_property.upper_bounds,
        output_constraint=output_constraint,
        property_name="strengthened " + original_property.property_name
    )


def conv1d_to_conv2d(network) -> NeuralNetwork:
    """
    Converts 1d convolutions to 2d convolutions by adding a virtual
    height dimension of size 1 to all kernels.
    ERAN only supports 2d convolutions, so conversion is necessary.
    This function also converts 2d inputs (Conv1d) to 3d inputs (Conv2d),
    even when no convolution is present.

    Currently only supports `torch.nn.Conv1d`, no pooling or batch normalization layers.

    :param network: The network whose 1d convolutions to convert.
    :return: A new network with 2d convolutions instead of 1d convolutions.
     When `network` doesn't contain 1d convolutions, `network` is returned directly.
    """
    modules = list(network.named_modules())[1:]  # first is network itself
    if (
        len(network.inputs_shape) != 2
        or all(not isinstance(module, torch.nn.Conv1d) for _, module in modules)
    ):
        return network
    new_network = deepcopy(network)
    if len(network.fixed_inputs_shape) == 2:
        new_network.fixed_inputs_shape = network.fixed_inputs_shape + (1,)
        new_network.mins = network.mins.unsqueeze(-1)
        new_network.maxes = network.maxes.unsqueeze(-1)
        new_network.means_inputs = network.means_inputs.unsqueeze(-1)
        new_network.ranges_inputs = network.ranges_inputs.unsqueeze(-1)
    for name, module in modules:
        if isinstance(module, torch.nn.Conv1d):
            new_conv = torch.nn.Conv2d(
                module.in_channels, module.out_channels,
                kernel_size=module.kernel_size + (1,),
                stride=module.stride + (1,),
                padding=module.padding + (0,),
                dilation=module.dilation + (1,),
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                device=module.weight.device,
                dtype=module.weight.dtype
            )
            with torch.no_grad():
                new_conv.weight.set_(module.weight.reshape_as(new_conv.weight))
                if module.bias is not None:
                    new_conv.bias.set_(module.bias)
            setattr(new_network, name, new_conv)
    return new_network


class ERAN(CounterexampleGenerator):
    """
    Wraps the verify_acasxu code from ERAN in a CounterexampleGenerator.
    """
    def __init__(self, use_acasxu_style: Union[str, bool] = 'auto',
                 single_counterexample: bool = False,
                 exit_mode: Union[str, float] = "optimal", **kwargs):
        """
        Creates a new ERAN with the given parameters for running verify_acasxu.

        See ``eran/tf_verify/verification_interface.verify_acasxu`` for valid parameters and default values.

        :param use_acasxu_style: Whether to use acasxu-style or "plain" verification, i.e. whether to use
            input bounds splitting or not.
            If set to 'auto', determined automatically depending
            on the dimensionality of the inputs.
            If more than 20 inputs are present, plain verification is used.
            Pass True to use acasxu-style verification in any case.
            Pass False to use plain verification.
        :param single_counterexample: Whether to return only a single counterexample if counterexamples were found.
            This does not speed up verification in any case, but is required for some repair backends.
        """
        self.use_acasxu_style = use_acasxu_style
        self.single_counterexample = single_counterexample
        self.exit_mode = exit_mode
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return 'ERAN'

    def find_counterexample(self, target_network: NeuralNetwork, target_property: Property) \
            -> Tuple[Optional[Sequence[Counterexample]], str]:
        start_time = time.monotonic()
        if target_property.input_constraint is not None:
            raise ValueError("ERAN currently can't handle Properties with additional input constraints.")

        # we need to convert the property to the format ERAN wants
        # input_bounds = [[(np.single(lb), np.single(ub)) for lb, ub in target_property.input_bounds(target_network)]]
        input_bounds = target_property.input_bounds(target_network)
        output_constraints = property_constraint_to_eran(
            target_property.output_constraint,
            target_network.num_outputs(),
            target_network.means_outputs.flatten().detach().cpu().numpy(),
            target_network.ranges_outputs.flatten().detach().cpu().numpy()
        )

        # convert model to ONNX and write it to a file to let ERAN convert it to it's internal representation
        # use a temporary directory instead of a named temporary file, because apparently on Windows
        # named temporary files can not be opened another time.
        with TemporaryDirectory() as temp_dir:
            onnx_file = os.path.join(temp_dir, 'network.onnx')
            converted_network = conv1d_to_conv2d(target_network)
            converted_network.onnx_export(onnx_file, disable_normalization=True, input_sample='batch')

            try:
                if (self.use_acasxu_style == 'auto' and len(input_bounds) <= 20) or self.use_acasxu_style is True:
                    counterexamples = verify_acasxu_style(
                        network_file=onnx_file,
                        means=target_network.means_inputs.flatten().detach().cpu().numpy(),
                        stds=target_network.ranges_inputs.flatten().detach().cpu().numpy(),
                        input_box=input_bounds, output_constraints=output_constraints,
                        exit_mode=self.exit_mode,
                        start_time=start_time,
                        **self.kwargs
                    )
                else:
                    counterexamples = verify_plain(
                        network_file=onnx_file,
                        means=target_network.means_inputs.flatten().detach().cpu().numpy(),
                        stds=target_network.ranges_inputs.flatten().detach().cpu().numpy(),
                        input_box=input_bounds, output_constraints=output_constraints,
                        exit_mode=self.exit_mode,
                        start_time=start_time,
                        **self.kwargs
                    )
            except Exception as e:
                # raise e
                warning(f"Exception while running ERAN for property {target_property.property_name}")
                traceback.print_exc()
                return None, f"ERROR: {e}"
        if counterexamples is None:
            return None, "UNKNOWN"
        elif len(counterexamples) == 0:
            return [], "VERIFIED"
        else:
            full_infos = [
                target_property.full_witness(
                    torch.tensor(inputs).unsqueeze(0),
                    target_network
                )
                for inputs, _, _ in counterexamples
            ]
            if self.single_counterexample:
                # select the counterexample with the largest violation
                sat_fn_values = [sat_value for _, _, sat_value, _ in full_infos]
                sat_fn_values = torch.stack(sat_fn_values)
                max_violation_index = torch.argmin(sat_fn_values)
                counterexamples = [counterexamples[max_violation_index]]
                full_infos = [full_infos[max_violation_index]]
            counterexamples = [
                Counterexample(
                    # "inputs" is a batched tensor, remove the batch dimension
                    inputs=inputs[0],
                    network_outputs=outputs, property_satisfaction=sat_val.item(),
                    property=eran_comparison_to_property(violated_constraint, target_property),
                    internal_violation=-value
                )
                for (_, violated_constraint, value), (inputs, outputs, sat_val, _)
                in zip(counterexamples, full_infos)
            ]
            return counterexamples, "NOT VERIFIED"

    def compute_bounds(self, target_network,
                       input_region: Union[Sequence[Tuple[float, float]], Tuple[Dict[int, float], Dict[int, float]]]) \
            -> Tuple[List[float], List[float]]:
        """
        Computes bounds on the network output.
        The first return value is the lower bounds, while
        the second return value is the upper bounds.

        :param target_network: The network to analyse.
        :param input_region: The input region to analyse.
        :return: lower bounds on network outputs, upper bounds on network outputs
        """
        if isinstance(input_region[0], Dict):
            input_lbs, input_ubs = input_region
            # use the code implemented in property to intersect with the network bounds
            void_property = Property(
                lower_bounds=input_lbs, upper_bounds=input_ubs,
                output_constraint=BoxConstraint(0, '<', 0)
            )
            input_bounds = void_property.input_bounds(target_network)
        else:
            input_bounds = input_region

        with TemporaryDirectory() as temp_dir:
            onnx_file = os.path.join(temp_dir, 'network.onnx')
            target_network.onnx_export(onnx_file, disable_normalization=True, input_sample='batch')

            bounds = compute_bounds(
                network_file=onnx_file,
                means=target_network.means_inputs.flatten().detach().cpu().numpy(),
                stds=target_network.ranges_inputs.flatten().detach().cpu().numpy(),
                input_box=input_bounds,
                **self.kwargs
            )
        return bounds
