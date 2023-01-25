from typing import Tuple, Optional, Sequence, Final
from logging import debug

import ray
import torch

import deep_opt
from deep_opt import NeuralNetwork, Property, CounterexampleAmount
from nn_repair.counterexamples import CounterexampleGenerator, Counterexample


class DeepOpt(CounterexampleGenerator):

    DEFAULT_COUNTEREXAMPLE_AMOUNT: Final[CounterexampleAmount] = \
        CounterexampleAmount.ALL_MINIMUMS | CounterexampleAmount.FOR_ALL_INPUT_BOUNDS

    def __init__(self, **kwargs):
        """
        Creates a new DeepOpt with the given parameters for running deep_opt.

        See ``deep_opt.optimize_by_property`` for valid parameters and default values.
        """
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return 'DeepOpt'

    def find_counterexample(self, target_network: NeuralNetwork, target_property: Property) \
            -> Tuple[Optional[Sequence[Counterexample]], str]:
        if not ray.is_initialized():
            debug("Ray not started when calling DeepOpt.find_counterexamples. Starting now. "
                  "Ray will not be shutdown automatically.")
            ray.init()

        status, counterexamples, _ = deep_opt.optimize_by_property(target_network, target_property, **self.kwargs)

        # DeepOpt may return a single counterexample if how_many_counterexamples is set to SINGLE
        # (for compatibility reasons)
        if not isinstance(counterexamples, list):
            counterexamples = [counterexamples]

        if status == "ERROR":
            return None, status
        elif status == "SAT":
            # turn the counterexamples returned by deep_opt into Counterexample objects
            # deep_opt returns flat inputs and outputs also for network with multidimensional inputs/outputs
            cx_inputs = [torch.tensor(cx[:target_network.num_inputs()]) for cx in counterexamples]
            full_infos = [target_property.full_witness(cx.unsqueeze(0), target_network) for cx in cx_inputs]
            counterexamples = [
                Counterexample(inputs_batched[0].detach(), outputs_internal, sat_val.item(), target_property)
                for inputs_batched, outputs_internal, sat_val, _ in full_infos
            ]
            return counterexamples, status
        elif status == "UNSAT":
            return [], status
