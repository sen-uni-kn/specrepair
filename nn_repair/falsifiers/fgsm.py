from typing import Tuple, Optional, Sequence

import torch

from deep_opt import NeuralNetwork, Property
from nn_repair.counterexamples import CounterexampleGenerator, Counterexample


class FastGradientSignMethod(CounterexampleGenerator):
    """
    Generates counterexamples/adversarial examples by walking into the direction
    of the signs of the gradients until hitting the bounds.

    This attack has been introduced in [Goodfellow2015]_.
    Here it is generalized in a way to work for arbitrary properties
    using the satisfaction function.

    .. [Goodfellow2015] Goodfellow, Ian J. / Shlens, Jonathon / Szegedy, Christian
     Explaining and Harnessing Adversarial Examples
     2015
     ICLR (Poster)
    """

    def __init__(self):
        """
        Creates a new FastGradientSignMethod (FGSM) for generating counterexamples.
        """
        pass

    @property
    def name(self) -> str:
        return 'FGSM'

    def find_counterexample(self, target_network: NeuralNetwork, target_property: Property) \
            -> Tuple[Optional[Sequence[Counterexample]], str]:
        if target_property.input_constraint is not None:
            raise ValueError("FGSM can not handle Properties with additional input constraints.")

        # start at the center point of the property
        bounds = target_property.input_bounds(target_network)
        lower = torch.tensor([lower for lower, _ in bounds])
        upper = torch.tensor([upper for _, upper in bounds])

        x = (lower + (upper-lower)/2).clone().detach().requires_grad_()
        # network also accepts flat inputs (but satisfaction function requires a batch dimension)
        violation = target_property.satisfaction_function(x.unsqueeze(0), target_network)
        violation.backward(inputs=[x])
        gradient_sign = -x.grad.sign()

        cx_candidate = torch.where(gradient_sign < 0, lower, upper)
        # reset places where the gradient is zero
        cx_candidate = gradient_sign.abs() * cx_candidate + (1 - gradient_sign.abs()) * x

        _, network_outputs, satisfaction, is_sat = \
            target_property.full_witness(cx_candidate.unsqueeze(0), target_network)
        if not is_sat:
            cx = Counterexample(inputs=cx_candidate.detach().numpy(),
                                network_outputs=network_outputs,
                                property_satisfaction=satisfaction.item(),
                                property=target_property)
            return (cx, ), 'Violated'
        else:
            return (), 'Unknown'
