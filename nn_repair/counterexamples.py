from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence, Union, Any
from dataclasses import dataclass

import numpy as np
import torch

from deep_opt import NeuralNetwork, Property


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Counterexample:
    """
    A counterexample with inputs that cause the violation of a property for some network
    and the property satisfaction as witness.
    `property_satisfaction` needs to be calculated using the original property.
    We also record the network outputs (potentially multiple) as further witness.

    Specifying a property allows strengthening the original property on the fly
    to deal with spurious counterexamples.
    If that is not desired, use the original property for the ``property`` attribute.

    Additionally, a counterexample generator may add the internal violation the generated counterexample has.
    This may also help to fix spurious counterexamples, for example such that only come up through serialisation.
    """
    inputs: Union[np.ndarray, torch.Tensor]
    network_outputs: Any
    property_satisfaction: float
    property: Property
    internal_violation: Optional[float] = None

    # the following method mostly silences a pytorch warning
    def inputs_as_tensor(self) -> torch.Tensor:
        if isinstance(self.inputs, torch.Tensor):
            return self.inputs
        else:
            return torch.tensor(self.inputs)


class CounterexampleGenerator(ABC):
    @abstractmethod
    def find_counterexample(self, target_network: NeuralNetwork, target_property: Property) \
            -> Tuple[Optional[Sequence[Counterexample]], str]:
        """
        Tries to find one or multiple counterexamples for which the given network violates the given property.
        If no counterexample could be found, this method returns an empty sequence.

        :param target_network: The network for which a counterexample should be tried to be found.
        :param target_property: The property to falsify for the given network.
        :return: Return an empty sequence if no counterexample could be generated.
            Otherwise, a sequence of one or multiple counterexamples is returned.
            Return None if an error occurred.
            As second return value, you may return a description of the exit status or details on the error if an error
            occurred.
            This string will be shown to the user or logged, but not used to interpret the result
            of the computation that is only done using the first return value.
            Return an empty string if you do not want to report a status message.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return 'CounterexampleGenerator'
