from typing import List, Tuple
from logging import info, warning, error

import torch

import gurobipy as gp
from gurobipy import GRB

from deep_opt import NeuralNetwork
import deep_opt.models.differentiably_approximatable_nn_modules as nn_layers
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.verifiers.eran import property_constraint_to_eran


class LinearProgrammingMinimalModificationRepairDelegate(RepairNetworkDelegateBase):
    """
    Implements the repair backend of [Goldberger2020]_ that repairs a network by modifying the weights
    of the last network layer trough linear programming.

    The repair backend minimizes the change in parameters while trying to fix the counterexamples
    by solving a linear program using Gurobi.

    .. [Goldberger2020] Ben Goldberger, Guy Katz, Yossi Adi, Joseph Keshet:
        Minimal Modifications of Deep Neural Networks using Verification. LPAR 2020: 260-278
    """

    def __init__(self, satisfaction_eps=1e-4, keep_all_counterexamples: bool = True):
        """
        Creates a new RepairNetworkDelegate that searches for the minimal
        modification of the parameters of the last layer using linear programming.
        """
        super().__init__(keep_all_counterexamples)
        self.sat_eps = satisfaction_eps

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.

        :return: The final status of the repair. Return SUCCESS if all counterexamples no longer
         violate the specification. Return FAILURE if this could not be archived.
         Return ERROR if any other error occurred from which recovery is possible, otherwise raise
         an exception.
        """
        last_layer = self.network[-1]
        if not isinstance(last_layer, nn_layers.Linear):
            raise ValueError(f"LinearProgrammingMinimalModificationRepairDelegate requires the last layer"
                             f"of the network to repair to be linear. However the last layer was: {last_layer}")

        # get the property output constraints in ERAN format
        # on the way check that the constraints are linear
        # ERAN format constraints are nested lists of 3-tuples:
        # - top level list: conjunction
        # - nested lists: disjunction
        # - 3-tuples (i, j, k):
        #   * j < 0: box constraint
        #     + j == -1: desired: out_i <= k
        #     + j == -2: desired out_i < k
        #     + j == -3: desired out_i >= k
        #     + j == -4: desired out_i > k
        #   * otherwise: desired out_i > out_j (k doesn't matter in this case)
        # We require that all nested lists (disjunction) have exactly one element (no disjunction).
        # We will duplicate the counterexample for each element in the outer list (conjunction)
        # for simplicity
        constraints: List[Tuple[torch.Tensor, Tuple[int, int, float]]] = []
        for prop, cx in self.unfolded_counterexamples:
            eran_format_constraint = property_constraint_to_eran(
                prop.output_constraint,
                self.network.num_outputs(),
                self.network.means_outputs.flatten().detach().numpy(),
                self.network.ranges_outputs.flatten().detach().numpy()
            )
            for or_list in eran_format_constraint:
                if len(or_list) != 1:
                    raise ValueError('LinearProgrammingMinimalModificationRepairDelegate does not'
                                     'handle non-linear properties (no disjunctions!).')
                constraints.append((cx, or_list[0]))

        info("Setting up LP model")
        lp_model = gp.Model("counterexample_fixing")
        # lp_model.Params.LogToConsole = 1
        # add variables for all weights to the model
        # we are using the same variable arrangement as in the weight tensor for the last layer
        # we are using two variables to encode the weight change
        # this way we can avoid using absolute values for the changes
        # we simply use one variable >= 0 which is added and one >= 0 which is subtracted
        weight_change_variables_plus = [[None] * last_layer.in_features for _ in range(last_layer.out_features)]
        weight_change_variables_minus = [[None] * last_layer.in_features for _ in range(last_layer.out_features)]
        total_weight_change = gp.LinExpr()
        for i in range(last_layer.out_features):
            for j in range(last_layer.in_features):
                weight_change_variable_plus = lp_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0, ub=float('inf'),
                    name=f'weight_change_plus_{i}_{j}'
                )
                weight_change_variable_minus = lp_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0, ub=float('inf'),
                    name=f'weight_change_minus_{i}_{j}'
                )
                weight_change_variables_plus[i][j] = weight_change_variable_plus
                weight_change_variables_minus[i][j] = weight_change_variable_minus
                total_weight_change += weight_change_variable_plus
                total_weight_change += weight_change_variable_minus
        lp_model.setObjective(total_weight_change, GRB.MINIMIZE)

        # the inputs to the last layer is what the network produces if we compute
        # all layers except for the last layer
        previous_layers = NeuralNetwork(
            mins=self.network.mins, maxes=self.network.maxes,
            means_inputs=self.network.means_inputs, ranges_inputs=self.network.ranges_inputs,
            # do not carry over output means and ranges. Output normalization is already handled in the
            # ERAN constraint format. Also we don't want to normalize intermediate layer output.
            # ---
            # take all layers except the last (first element of modules is network itself)
            modules=list(self.network.modules())[1:-1],
            inputs_shape=self.network.inputs_shape
        )

        def get_output_computation(out_index, last_layer_inputs_):
            # set up the formula for computing a certain output
            out = gp.LinExpr()
            for in_index in range(len(last_layer_inputs_)):
                # (weight + weight_change_plus - weight_change_minus) * input
                # = weight * input + weight_change_plus * input - weight_change_minus * input
                out += last_layer.weight[out_index, in_index].item() * last_layer_inputs[in_index].item() \
                       + weight_change_variables_plus[out_index][in_index] * last_layer_inputs[in_index].item() \
                       - weight_change_variables_minus[out_index][in_index] * last_layer_inputs[in_index].item()
            out += last_layer.bias[out_index]
            return out

        # encode fixing the counterexamples as constraints
        for cx, (i, j, k) in constraints:
            # compute the input to the last layer
            last_layer_inputs = previous_layers(cx)
            # add constraints for the relevant outputs
            # the other outputs doesn't matter, we don't have to encode those
            if j < 0:
                out_i = get_output_computation(i, last_layer_inputs)
                # box constraint (out_i >=< k)
                assert j >= -4
                if j == -1 or j == -2:  # handing is the same due to use of sat_eps
                    lp_model.addConstr(out_i <= float(k) - self.sat_eps)
                elif j == -3 or j == -4:
                    lp_model.addConstr(out_i >= float(k) + self.sat_eps)
            else:
                out_i = get_output_computation(i, last_layer_inputs)
                out_j = get_output_computation(j, last_layer_inputs)
                lp_model.addConstr(out_i - out_j >= self.sat_eps + float(k))

        lp_model.update()
        info(f'Starting LP-solving. \n'
             f'Number of variables: {lp_model.NumVars}\n'
             f'Number of constraints: {lp_model.NumConstrs}')
        lp_model.optimize()

        if lp_model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            # update the network parameters
            for i in range(last_layer.out_features):
                for j in range(last_layer.in_features):
                    last_layer.weight.data[i, j] = last_layer.weight.data[i, j] \
                                                   + weight_change_variables_plus[i][j].x \
                                                   - weight_change_variables_minus[i][j].x
            info(f'LP-solving finished.\n'
                 f'Weight change: {lp_model.ObjVal}\n')
            if self.count_violations() > 0:  # feasibility may be spurious, relevant if sat_eps = 0
                error(f'Not all counterexamples fixed after successfully solving LP (spurious feasibility).\n'
                      f'Satisfaction_eps: {self.sat_eps}\n'
                      f'Model status: {lp_model.Status}\n'
                      f'Model: {lp_model.display()}')
                return RepairStatus.ERROR
            return RepairStatus.SUCCESS
        elif lp_model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            warning('LP-model is infeasible. Counterexamples can not be fixed by modifying last layer weights.\n'
                    'Network has not been modified.')
            return RepairStatus.FAILURE
        else:
            warning(f'LP-solving failed with status code: {lp_model.Status}. Model: {lp_model}\n'
                    'Network has not been modified.')
            return RepairStatus.FAILURE
