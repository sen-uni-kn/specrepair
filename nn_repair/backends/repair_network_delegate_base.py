import math
from typing import Sequence, List, Dict, Tuple
from abc import ABC

import numpy as np
import torch

from deep_opt import NeuralNetwork, Property
from nn_repair.repair import RepairNetworkDelegate
from nn_repair.backends import ConstraintFunctionFactory


class RepairNetworkDelegateBase(RepairNetworkDelegate, ABC):
    """
    Abstract base class for RepairNetworkDelegates that
    provides default implementations.
    """

    def __init__(self, keep_all_counterexamples: bool = True):
        self.keep_counterexamples = keep_all_counterexamples

        self._network = None
        self._counterexamples: Dict[Property, List[torch.Tensor]] = {}

    def with_specification(self, specification: Sequence[Property]):
        """
        Prepares the RepairNetworkDelegate to repair counterexamples
        to the given specification.

        This resets all previously stored counterexamples.

        :param specification: The specification to use.
        """
        self._counterexamples = dict((p, []) for p in specification)

    def register_strengthened_property(self, strengthened_property: Property, original_property: Property):
        """
        Register another property that is a strengthened version of an already
        registered property in order to prepare the RepairNetworkDelegate to
        repair counterexamples to this strengthened property.

        Strengthened properties may be registered multiple times.

        :param strengthened_property: The strengthened version of the original_property
          (typically allows to handle spurious counterexamples).
        :param original_property: The original property of which strengthened_property is a strengthened version.
        """
        if strengthened_property not in self._counterexamples:
            self._counterexamples[strengthened_property] = []

    @property
    def network(self) -> NeuralNetwork:
        """
        The network to repair, if already set.

        :raises ValueError: If not network was set yet.
        """
        if self._network is None:
            raise ValueError("No network set yet.")
        return self._network

    @network.setter
    def network(self, network: NeuralNetwork):
        """
        Sets the network that should be repaired.
        """
        self._network = network

    def new_counterexample(self, prop: Property, network_inputs: torch.Tensor):
        """
        Updates the counterexamples used for repair with the new counterexample for the given property.

        This method is called repeatedly before repair is invoked.

        :param prop: The property for which a new counterexample was found.
        :param network_inputs: The input values for which the current network violates the specification.
        """
        tensor_inputs = network_inputs.detach()
        if self.keep_counterexamples:
            self._counterexamples[prop].append(tensor_inputs)
        else:
            self._counterexamples[prop] = [tensor_inputs]

    def property_satisfied(self, prop: Property):
        """
        Calls of this method inform a RepairNetworkDelegate that a specification is satisfied with the
        current network. This may be useful to clean up counterexamples if this is required by your method.
        :param prop: The property that is satisfied by the network.
        """
        if not self.keep_counterexamples:
            self._counterexamples[prop] = []

    @property
    def counterexamples(self) -> Dict[Property, List[torch.Tensor]]:
        """
        The currently stored (former) counterexamples, per property.

        Despite being called counterexamples, these network inputs
        do not necessarily lead to a property violation.
        In particular, after repair has been called, these counterexamples
        should have been fixed.

        **The inputs stored in this property are inputs that once caused
        a property violation and potentially still do so.**
        """
        return self._counterexamples

    @property
    def unfolded_counterexamples(self) -> List[Tuple[Property, torch.Tensor]]:
        """
        The currently stored (former) counterexamples, not organised per property.
        The returned list will contain duplicate properties whenever there are multiple
        counterexamples registered for one property.

        See the counterexamples property for more details on the returned
        (potentially former) counterexamples.

        :return: A list of counterexamples with the properties they violate
         or once violated as first tuple element and the respective network input
         as second input.
        """
        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = []
        for p, cs in self._counterexamples.items():
            for c in cs:
                unfolded_counterexamples.append((p, c))
        return unfolded_counterexamples

    def count_violations(self, unfolded_counterexamples=None) -> int:
        """
        Counts the number of stored (potential) counterexamples
        that currently violate the specification.

        :param unfolded_counterexamples: The unfolded counterexamples
         of which the violations should be counted.
         If None the value of the unfolded_counterexamples property is used.
        :return: The number of recorded violations.
        """
        if unfolded_counterexamples is None:
            unfolded_counterexamples = self.unfolded_counterexamples
        # uses True ~> 1, False ~> 0
        return sum(not p.property_satisfied(c.unsqueeze(0), self._network)
                   for p, c in unfolded_counterexamples)

    def violation_stats(self) -> str:
        """
        Create a log message that gives details on the violation of the properties with the counterexamples.

        The implementation in RepairNetworkDelegateBase returns a tabular overview
        of the stored properties and the overall violation trough it's counterexamples.

        The violation is calculated using the ``_calc_violation_for_stats`` method
        that by default returns the l1 violation (negated property violation
        calculated using the properties satisfaction_function
        if the satisfaction is negative, otherwise zero).

        After the table also a text block visualising the counterexamples and their violations
        is shown ("counterexample art")

        :return: A text giving statistics about the violation of the specification with
        the counterexamples that will be used in the next invocation of repair.
        """
        # Note: here without satisfaction epsilon
        constr_factory = ConstraintFunctionFactory().with_network(self.network)
        stats = []
        prop_name_max_len = 0
        violation_max_len = 0
        for prop in self.counterexamples.keys():
            constr_factory.with_property(prop)
            overall_violation = sum(self._calc_violation_for_stats(prop, c, constr_factory)
                                    for c in self.counterexamples[prop])
            stats.append((prop.property_name, overall_violation))

            prop_name_max_len = max(prop_name_max_len, len(prop.property_name))
            violation_max_len = max(violation_max_len, len(f"{overall_violation:.4f}"))

        mean_violation = 0
        if len(self.counterexamples) > 3:
            mean_violation = sum(s[1] for s in stats) / len(self.counterexamples)
            # prop_name_max_len = max(prop_name_max_len, 4)  # len("mean") == 4  # see below; 9 > 4
            violation_max_len = max(violation_max_len, len(f"{mean_violation:.4f}"))

        prop_name_max_len = max(prop_name_max_len, 12)  # len("Property    ") == 12
        violation_max_len = max(violation_max_len, 13)  # len("Violation    ") == 13
        stats_string = "Property    " + " " * (prop_name_max_len - 12) + "  Violation"
        stats_string += "\n" + "-" * (prop_name_max_len + violation_max_len + 2) + "\n"  # +2 for two spaces
        stats_string += "\n".join([p + " " * (prop_name_max_len - len(p)) + "  "
                                   + (f"{v:.4f}" if v != 0 else "---") for p, v in stats])
        if len(self.counterexamples) > 3:
            stats_string += "\n" + "-" * (prop_name_max_len + violation_max_len + 2)  # +2 for two spaces
            stats_string += "\nMEAN" + " " * (prop_name_max_len - 4) + "  " + str(mean_violation)
        return stats_string

    def counterexample_art(self, unfolded_counterexamples=None, octiles=None, max_width=50) -> Tuple[str, List[float]]:
        """
        Produces a textual/visual overview of the violations
        of all counterexamples in a text block like so:

            *~0#0+0--0,**+-0

        For 16 counterexamples. This visualisation is useful for tracking
        which counterexamples are satisfied.

        The different symbols track the magnitude of the violation.
        ``0`` indicates no violation. The remaining symbols indicate in what quantile
        of the violations a violation resides.

         * ``#`` Top quantile: 87.5% - 100%
         * ``*`` 75% - 87.5%
         * ``+`` 62.5% - 75%
         * ``~`` 50% - 62.5%
         * ``-`` 37.5% - 50%
         * ``:`` 25% - 37.5%
         * ``,`` 12.5% - 25%
         * ``.`` Bottom quantile: 0% - 12.5%

        The symbols are organized in a text block, such that
        the block is rectangular, but not wider than 50 characters

            **~0#0+0--0,**+-0~##*..,-0**~0#0+0--0,**+-0~##*..,
            -0**~0#0+0--0,**+-0~##*..,-0**~0#0+0--0,**+-0~##*.
            .,**~0#0+0--0,**+-0~##*..,-0**~0#0+0--0,**+-0~##*0
            ..,.,**~0#0+0--0,**+-0~##*..,-0**~0#0+0--0,**+-

        This method also uses the _calc_violation_for_stats method to compute
        violations. By default that method returns the l1 violation.

        :param unfolded_counterexamples: The counterexamples to consider.
         If None, the unfolded_counterexamples property is used.
        :param octiles: A precalculated scale for choosing the violation symbols.
         Needs to have length 7. Use for example the second argument returned by this method.
         If None, octiles are calculated in this method.
        :param max_width: Maximum number of symbols in one line of the created text box.
        :return A text box string visualising the violations and an octiles list, that
         can be used as a scale for further calls.
        """
        if unfolded_counterexamples is None:
            unfolded_counterexamples = self.unfolded_counterexamples

        # Note: here without satisfaction epsilon
        constr_factory = ConstraintFunctionFactory().with_network(self.network)
        violations = []
        for prop, cx in unfolded_counterexamples:
            constr_factory.with_property(prop)
            violations.append(self._calc_violation_for_stats(prop, cx, constr_factory))

        violations = np.array(violations)
        if octiles is None:
            octiles = [np.quantile(violations, 0.125 * i) for i in range(1, 8)]
        symbols = [self._counterexample_art_get_symbol(v, octiles) for v in violations]

        # arrange in a good looking text box
        if len(violations) < 10:
            lines = 1
            width = len(violations)
        else:
            # one line is roughly twice as high as width.
            # This has to be taken into account when calculating the golden rectangle weight
            # it is taken into account    | here, because usually we would see a two there
            golden_ratio_width = np.sqrt((len(symbols)) / (1 + np.sqrt(5)))
            lines = np.ceil(golden_ratio_width)
            width = np.ceil(len(symbols) / lines)
            if width > max_width:
                width = max_width
                lines = np.ceil(len(symbols) / width)
            lines = int(lines)
            width = int(width)

        text_box = u'    ┌─' + ('─' * width) + '─┐\n'
        # text_box += '    │ ' + (' ' * width) + ' │\n'
        symbol_iter = iter(symbols)
        for _ in range(lines):
            text_box += u'    │ '
            text_box += ''.join([next(symbol_iter, ' ') for _ in range(width)])
            text_box += u' │\n'
        # text_box += '    │ ' + (' ' * width) + ' │\n'
        text_box += u'    └─' + ('─' * width) + '─┘\n'
        return text_box, octiles

    def _counterexample_art_get_symbol(self, violation, octiles):
        """
        Get the symbol for a violation based on the octiles (quantiles with 1/8 steps).
        :param violation: The violation to visualize
        :param octiles: List of octiles. Calculate like this:

            [np.quantile(violations, 0.125 * i) for i in range(1, 8)]
        :return: One of the characters 0, ., ,, :, -, ~, +, * and # (see counterexample_art for more details)
        """
        if violation <= 0:
            return '0'

        if math.isnan(violation) or math.isinf(violation):
            return str(violation)

        assert len(octiles) == 7
        symbols = ['.', ',', ':', '-', '~', '+', '*']
        for symbol, octile in zip(symbols, octiles):
            if violation <= octile:
                return symbol
        return '#'  # larger than last octile (87.5%)

    def _calc_violation_for_stats(self, prop: Property, cx: torch.Tensor,
                                  constraint_factory: ConstraintFunctionFactory) -> float:
        """
        Calculate the violation of the given property with the given (potential) counterexample
        for logging purposes (violation_stats method).

        The default implementation returns the violation using the l1 penalty,
        so just the (negated) property satisfaction value, if this value is negative.

        The returned value needs to be positive or zero, if the property is not
        violated for the given counterexample.

        :param prop: The property of which the violation should be calculated.
        :param cx: The counterexample of which the violation should be calculated.
        :param constraint_factory: A constraint factory that may be helpful
         for calculating the violation. The constraint factory already has
         prop registered as property (with_property was already called).
        :return: A non-negative float value measuring violation of the given property
         by the given network input (counterexample).
        """
        return constraint_factory.create_l1_penalty(cx)().item()
