from collections import defaultdict
from functools import partial
from logging import warning, error, info
from typing import Union, List, Tuple, Dict, Callable

import torch
import torch.nn.functional as F

from deep_opt.models.property import OutputConstraint, ConstraintAnd, ConstraintOr, BoxConstraint, ExtremumConstraint, \
    MultiOutputExtremumConstraint
import deep_opt.models.differentiably_approximatable_nn_modules as nn_layers
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.utils.shift_property import ShiftedOutputConstraint


def _stacked_loss(loss_functions_):
    stacked = torch.stack([violation() for violation in loss_functions_])
    return torch.sum(stacked, dim=0)


class NeuronFixationRepairDelegate(RepairNetworkDelegateBase):
    """
    Implements the repair backend of [Dong2020]_ that is based on identifying influential
    neurons (that have a high gradient value for a violation loss function) and setting
    their output value to a constant.

    Setting the output of a neuron to a constant (fixating the output) corresponds to
    setting all ingoing weights of the neuron to zero and the bias to the desired output value.

    .. [Dong2020] Guoliang Dong, Jun Sun, Jingyi Wang, Xinyu Wang, Ting Dai:
        Towards Repairing Neural Networks Correctly. CoRR abs/2012.01872 (2020)
    """

    def __init__(self, maximum_fixations: Union[int, float] = 0.05, maximum_fixations_per_neuron: int = 50,
                 step_size=0.35):
        """
        Creates a new RepairNetworkDelegate that iteratively fixates the output
        of individual neurons (sets it to a constant).

        The created backend is intended for one time use. It maintains a counter how often neurons have
        been modified, that is never reset. This counter determines whether repair has failed.

        :param maximum_fixations: The maximum number of fixations that are allowed for fixing the counterexamples.
         Values > 1 give concrete number of neurons that may be modified. If the value is > 0 and <= 1,
         it is used as the fraction of neurons of a network that may be fixated for fixing the counterexamples.
         This is the :math:`\\alpha` parameter of the counterexample fixing algorithm.
        :param maximum_fixations_per_neuron: The largest permitted number of times the output of one neuron may
         be fixated for fixing counterexamples.
         This is the :math:`\\beta` parameter of the counterexample fixing algorithm.
        :param step_size: The size of the step to perform when fixing the output of a neuron to a value.
         This is the :math:`\\eta` parameter of the counterexample fixing algorithm.
        """
        super().__init__(keep_all_counterexamples=False)

        assert maximum_fixations > 0 and (int(maximum_fixations) == maximum_fixations or maximum_fixations <= 1)
        assert maximum_fixations_per_neuron > 0
        assert step_size > 0
        self.alpha = maximum_fixations
        self.beta = maximum_fixations_per_neuron
        self.eta = step_size

        # will count how often we have changed the output of a neuron to determine when the repair has failed
        # neurons will be indexed as 2-tuples where the first element gives the layer which contains
        # the neuron and the second element gives the index of the neuron in that layer (e.g. index in the bias vector)
        # this counter is not reset during the repair
        self.neuron_modification_counters = defaultdict(int)

    @property
    def number_of_modified_neurons(self) -> int:
        return len([i for i in self.neuron_modification_counters.values() if i > 0])

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.

        :return: The final status of the repair. Return SUCCESS if all counterexamples no longer
         violate the specification. Return FAILURE if this could not be archived.
         Return ERROR if any other error occurred from which recovery is possible, otherwise raise
         an exception.
        """
        if not self.network.is_fully_connected():
            raise ValueError("NeuronFixationRepairDelegate requires fully connected networks. ")

        num_neurons = sum(layer.out_features for layer in self.network if isinstance(layer, nn_layers.Linear))
        effective_alpha = self.alpha if self.alpha > 1 else int(self.alpha * num_neurons)
        info(f"Starting neuron fixation repair. "
             f"Permitted number of neurons to modify: {effective_alpha} ({effective_alpha/num_neurons*100:.0f}%)")

        linear_layers = [layer for layer in self.network if isinstance(layer, nn_layers.Linear)]
        # torch backward hook allows accessing gradients wrt the output of a layer
        layer_output_gradients: Dict[nn_layers.Linear, torch.Tensor] = {}
        layer_outputs: Dict[nn_layers.Linear, torch.Tensor] = {}

        def store_gradient_hook(module, _, grad_outputs):
            layer_output_gradients[module] = grad_outputs[0]

        def store_output_hook(module, _, outputs):
            layer_outputs[module] = outputs

        # may only be called after .backward has been called
        # selects the neuron with the largest gradient, that has not been modified to often yet
        def select_neuron() -> Tuple[int, int]:
            max_grad = -float('inf')
            max_grad_index = None
            for i, layer in enumerate(linear_layers):
                layer_grad = layer_output_gradients[layer][0]  # grad has a batch dimension
                sorted_values, sorted_indices = layer_grad.sort(descending=True)
                for val, j_tensor in zip(sorted_values, sorted_indices):
                    j = j_tensor.item()
                    past_modifications = self.neuron_modification_counters[(i, j)]
                    if val > max_grad and past_modifications < self.beta:
                        max_grad = val
                        max_grad_index = (i, j)
            return max_grad_index

        handles = []
        try:
            for layer in linear_layers:
                backward_handle = layer.register_full_backward_hook(store_gradient_hook)
                forward_handle = layer.register_forward_hook(store_output_hook)
                handles.append(backward_handle)
                handles.append(forward_handle)

            # The original algorithm of Dong et al. is designed for fixing one counterexample
            # of one property at a time. To extend it to a full specification of multiple
            # properties, we have to iterate the algorithm for each counterexample
            counterexamples_processed_counter = 0
            num_counterexamples = len(self.unfolded_counterexamples)
            for prop, counterexample_inputs in self.unfolded_counterexamples:
                info(f"Processing counterexample {counterexamples_processed_counter+1} of {num_counterexamples}")
                # setup the loss function that indicates in which direction the different outputs should change
                # first step is to find out in what direction the outputs should change
                # and whether to encode them "cross-entropy-style" or "MSE-style"
                # if we do not set requires_grad to True, the backward hook for the first layer is not called
                counterexample_inputs = counterexample_inputs.requires_grad_(True)
                loss_function = self._get_dong_et_al_violation_function(counterexample_inputs, prop.output_constraint)
                while self.number_of_modified_neurons < effective_alpha:
                    if prop.property_satisfied(counterexample_inputs.unsqueeze(0), self.network):
                        counterexamples_processed_counter += 1
                        info(f"Counterexample {counterexamples_processed_counter} fixed "
                             f"({counterexamples_processed_counter/num_counterexamples*100:3.0f}%)")
                        break

                    # 1. compute the gradients of the loss function wrt. the neuron outputs
                    self.network.zero_grad()
                    loss = loss_function()  # will cause the values to be stored (forward hook)
                    loss.backward()  # will cause the gradients to be stored (backward hook)

                    # 2. select the neuron with the largest gradient that has not been modified too often
                    selection = select_neuron()
                    if selection is None:
                        error("No modifiable neurons remaining. ")
                        return RepairStatus.ERROR
                    layer_i, neuron_i = selection
                    layer = linear_layers[layer_i]

                    # 3. update the output of the neuron
                    # intermediate network outputs have a batch dimension
                    neuron_output = layer_outputs[layer][0, neuron_i]  # zeta
                    neuron_grad = layer_output_gradients[layer][0, neuron_i]  # nabla
                    new_output = neuron_output - self.eta * neuron_grad

                    # now set the output. That is a bit tricky, since we can only update the parameters
                    # fortunately setting the output to a value of a linear neuron simply means zeroing all weights
                    # and setting the bias to the desired value
                    layer.weight.data[neuron_i] = torch.zeros(layer.weight[neuron_i].shape)
                    layer.bias.data[neuron_i] = new_output

                    # 4. update the modification counter
                    self.neuron_modification_counters[(layer_i, neuron_i)] += 1
                    info(f'Fixed output of neuron ({layer_i}, {neuron_i}) to {new_output.item():.4f} '
                         f'(modification counter: {self.neuron_modification_counters[(layer_i, neuron_i)]}). '
                         f'{self.number_of_modified_neurons} neurons modified in total.')
                else:
                    # maximum number of neuron modifications exhausted
                    warning("Maximum number of neurons modified. Repair failed.")
                    return RepairStatus.FAILURE
        finally:
            for handle in handles:
                handle.remove()
        info(f'All counterexamples fixed. '
             f'Performed {sum(self.neuron_modification_counters.values())} modifications in total.')
        return RepairStatus.SUCCESS

    def _get_dong_et_al_violation_function(self, cx: torch.Tensor, constraint: OutputConstraint) \
            -> Callable[[], torch.Tensor]:
        """
        Returns a scalar function measuring violation of the given constraint for the given counterexample (cx)
        """
        if isinstance(constraint, ShiftedOutputConstraint):
            constraint = constraint.original

        if isinstance(constraint, ConstraintAnd) or isinstance(constraint, ConstraintOr):
            # doesn't matter which, just sum all the losses
            # for both and and or constraints it makes sense
            # to increase/decrease all relevant outputs for the counterexamples
            loss_functions = [self._get_dong_et_al_violation_function(cx, constr)
                              for constr in constraint.constraints]
            return partial(_stacked_loss, loss_functions)
        elif isinstance(constraint, BoxConstraint):
            # create "MSE-style" losses
            # this means we simply want to decrease/increase the respective output, no matter what the bound is
            out_index = constraint.output_index
            if constraint.less_than:  # <= and < => decrease this output
                def lt_loss():
                    return self.network(cx)[out_index]
                return lt_loss
            else:  # >= and > => increase this output
                def gt_loss():
                    return -self.network(cx)[out_index]
                return gt_loss
        elif isinstance(constraint, ExtremumConstraint) or isinstance(constraint, MultiOutputExtremumConstraint):
            # create "cross-entropy-style" losses
            out_indices: List[int]
            equals_or_in: bool
            # both constraint types have the field maximum
            if isinstance(constraint, ExtremumConstraint):
                out_indices = [constraint.output_index]
                equals_or_in = constraint.equals
            else:
                out_indices = constraint.output_indices
                equals_or_in = constraint.contained_in

            # first step: identify whether the outputs in out_indices should be increased or decreased
            if equals_or_in:
                # the maximum/minimum output should be among the listed outputs
                # maximum: => increase the listed outputs relatively to the others
                # minimum: => decrease
                increase = constraint.maximum
            else:
                # the maximum/minimum output should *not* be among the listed outputs
                # maximum: => decrease the listed outputs relatively to the others
                # minimum: => increase
                increase = not constraint.maximum

            # now create the loss function
            def ce_style_loss():
                outputs = self.network(cx)
                outputs = F.softmax(outputs, dim=0)
                factor = -1 if increase else 1
                return factor * torch.sum(outputs[out_indices])
            return ce_style_loss
        else:
            raise ValueError(f"Unsupported constraint: {constraint}.\n"
                             f"Either use ConstraintAnd, ConstraintOr, BoxConstraint, ExtremumConstraint, "
                             f"or MultiOutputExtremumConstraint.")
