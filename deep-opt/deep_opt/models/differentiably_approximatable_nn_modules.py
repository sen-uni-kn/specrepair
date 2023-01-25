from typing import Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum

import torch


class ApproximationDirection(Enum):
    """
    Approximation directions for DiffApproxModule.
    Note that these names are arbitrary and only serve to identify the one and the other approximation direction.
    Direction main is not somehow more important or prioritized over the complementary direction. <br>
    See DiffApproxModule for more details.
    """
    MAIN = 0
    COMPLEMENTARY = 1


class DiffApproxModule(torch.nn.Module, ABC):
    """
    Objects of this class are neural network modules
    that can be approximated arbitrarily accurately from two complementary sides
    with functions that are differentiable everywhere.<br>
    An example of such a function is ReLU. ReLU is not differentiable everywhere,
    still it can be approximated with SoftPlus and SiLU from above and below.
    For more complex functions (such as neural networks with multiple layers)
    an approximation might not necessarily always be above or below the original
    functions output (if we can at all talk about this, considering multidimensional outputs),
    but what's important is that the approximations to the function close in on the original function
    from two complementary directions. The approximating functions form a closed "band" around the original
    function. If we make tighter and tighter approximations, then there is no direction is space that we can
    "miss" with the approximations. <br>
    The two directions are called main and complementary direction in this class to distinguish them,
    but none of them is prioritized in any way. You can choose any direction to be the "main" or the "complementary"
    direction in your implementation (they should however not change dynamically).

    Activation functions or other transformations that are already differentiable everywhere can also
    be `DiffApproxModule` instances. In this case these functions are their own approximations.
    Have a look at the DiffModule class to create such objects. The needs_approximation
    method determines whether a DiffApproxModule object actually needs to be approximated
    or if it is itself differentiable everywhere.

    This abstract class has two abstract methods that have to be overwritten:
    _main_approx_impl and _complementary_approx_impl.
    These methods create an approximating differentiable module
    that approximates this modules transformation function with a certain tightness.
    Tightness is always given as a natural number (an integer >= 1).
    There is no guideline how tight the approximation for a certain value should actually be.
    However it needs to hold that as the tightness goes towards infinity,
    the difference between the approximation and the actual activation function needs to vanish.
    More details can be found in those methods doc strings.<br>

    Additionally to these methods also an implementation of forward needs to be provided.
    """

    def get_approximation_from_direction(self, tightness: int, direction: ApproximationDirection):
        """
        Returns an under or over approximation of this module.
        Have a look at get_under_approximation and get_over_approximation for more details.
        """
        if direction is ApproximationDirection.MAIN:
            return self.get_approximation(tightness)
        else:
            return self.get_co_approximation(tightness)

    def get_approximation(self, tightness: int) -> 'DiffModule':
        """
        Returns an approximation of this module with a certain tightness from the "main" side.
        The values of the returned module are always smaller than or equals to the
        outputs of this module.

        :param tightness: A natural number that specifies the tightness of the approximation.
        There is no guarantee how tight exactly
        the approximation will be, but if tightness goes to infinity, the returned approximation converges
        to the actual transformation function. Too large values of tightness may lead to numerical instabilities
        as the returned approximation tends towards a non-differentiable function.
        :return: A differentiable module that approximates this module from below.
        """
        if tightness < 1 or not isinstance(tightness, int):
            raise Exception("Tightness needs to be a natural number.")
        return self._main_approx_impl(tightness)

    @abstractmethod
    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        """
        This methods returns an approximating differentiable module
        that approximates the transformation function of this module with a certain tightness from the "main" side.
        To be an under-approximation it needs to hold that for any inputs to the under-approximation the returned output
        is always smaller than or equals to the output of the approximated transformation function. These outputs should
        in general get closer and closer to the approximated values as the tightness gets larger.
        :param tightness: The tightness that the returned approximation should have
        :return: A module that under-approximates this module.
        """
        raise NotImplementedError()

    def get_co_approximation(self, tightness: int) -> 'DiffModule':
        """
        Returns an approximation of this module with a certain tightness from the "complementary" side.
        The values of the returned module are always larger than or equals to the
        outputs of this module.

        :param tightness: A natural number that specifies the tightness of the approximation.
        There is no guarantee how tight exactly
        the approximation will be, but if tightness goes to infinity, the returned approximation converges
        to the actual transformation function. To large values of tightness may lead to numerical instabilities
        as the returned approximation tends towards a non-differentiable function.
        :return: A differentiable module that approximates this module from above.
        """
        if tightness < 1 or not isinstance(tightness, int):
            raise Exception("Tightness needs to be a natural number.")
        return self._complementary_approx_impl(tightness)

    @abstractmethod
    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        """
        This method returns an approximating differentiable module that approximates the transformation function
        of this module with a certain tightness from the "complementary" side.
        To be an over-approximation it needs to hold that for any inputs to the over-approximation the returned output
        is always larger than or equals to the output of the approximated transformation function. These outputs should
        in general get closer and closer to the approximated values as the tightness gets larger.
        :param tightness: The tightness that the returned approximation should have
        :return: A module that over-approximates this module.
        """
        raise NotImplementedError()

    def needs_approximation(self):
        """
        Whether this DiffApproxModule instance needs to be approximated or not.
        Concretely any module that is already differentiable does not need to be approximated.
        :return: True by default
        """
        return True

    @abstractmethod
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        """
        The tensor shape of the inputs of this module. Return None if your module accepts inputs of any shape.
        Dimensions that can have any size should be marked using the -1 wildcard.

        The first (batch) dimension is not contained in the inputs shape.
        :return: The tensor shape of the inputs to this module or None if inputs of any shape are accepted.
        """
        raise NotImplementedError()

    @abstractmethod
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        """
        The tensor shape of the outputs of this module.
        Return None if the output shape depends entirely on the input shape.
        If only some dimension sizes of the output shape depend on the inputs,
        these dimensions should be marked using -1.

        The first (batch) dimension is not contained in the outputs shape.
        :return: The tensor shape of the outputs of this module or
        None if this depends (entirely) on the number of inputs.
        """
        raise NotImplementedError()


class DiffModule(DiffApproxModule, ABC):
    """
    A class representing neural network transformation modules that are differentiable everywhere.
    Such functions are seen as perfect differentiable approximations of themselves.
    Calls to get_under_approximation and get_over_approximations will hence return
    the very same object they are called on.

    This class provides implementations of _under_approx_impl and _over_approx_impl.
    Implementing classes need to provide an implementation of call still.
    """

    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        return self

    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        return self

    def needs_approximation(self):
        """
        Whether this DiffApproxModule instance needs to be approximated or not.
        A differentiable module needs doesn't need to be approximated.
        :return: False
        """
        return False


class AnyInputsShapeModule(DiffApproxModule, ABC):
    """
    A module which accepts any input shape.

    This class provides implementations of inputs_shape() and outputs_shape() that return None by default.
    """
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        return None

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        return None


class ElementwiseModule(AnyInputsShapeModule, ABC):
    """
    A layer that applies an element wise function and can hence accept inputs with any shape.

    This class provides implementations of inputs_shape() and outputs_shape() that return None by default.
    """


class DiffApproximatingModule(DiffModule, ABC):
    """
    Abstract base class for parametric activation functions that can approximate
    other activation functions with a certain tightness.<br>
    """

    def __init__(self, tightness: int, **kwargs):
        """
        Creates a new approximating activation module.
        :param tightness: How close the approximation should be.
        """
        super().__init__(**kwargs)
        self.tightness = tightness


class ParametricSiLU(ElementwiseModule, DiffApproximatingModule, torch.nn.SiLU):
    """
    An activation module with the SiLU function, an under-approximation to ReLU:

    .. math:: y = \\frac{x}{1+e^{-tx}}
    where :math:`t` is the tightness parameter that is passed to the constructor.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # use torch.mul to silence the type warning that comes up when using *
        return inputs * torch.sigmoid(torch.mul(self.tightness, inputs))


class ParametricSoftplus(ElementwiseModule, DiffApproximatingModule, torch.nn.Softplus):
    """
    An activation module with the SoftPlus function, an over-approximation to ReLU:

    .. math:: y = \\frac{\\ln(1 + e^{kx})}{k}
    where :math:`t` is the tightness parameter that is passed to the constructor.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        val = torch.nn.Softplus.forward(self, torch.mul(self.tightness, inputs))
        return torch.div(val, self.tightness)


class ReLU(ElementwiseModule, DiffApproxModule, torch.nn.ReLU):
    """
    A ReLU activation module that can be approximated
    with the differentiable functions SiLU and SoftPlus.

    The "main" approximation returns an under-approximation with SiLU.
    The "complementary" approximation returns an over-approximation with SoftPlus.
    """

    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        # this is an under-approximation
        return ParametricSiLU(tightness)

    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        # this is an over-approximation
        return ParametricSoftplus(tightness)


class MaxPool1d(DiffApproxModule, torch.nn.MaxPool1d):
    """
    The one-dimensional Max Pooling operation (see torch.nn.MaxPool1d).
    The approximation of this module is based on expressing the max operation of two arguments using ReLU:

    .. math::
    \\max(x,y) = ReLU(x-y) + y

    this function can be chained to obtain an arbitrary argument max operation from ReLU.
    The "main" approximation is then build by replacing ReLU with SiLU in the above formula.
    The "complementary" approximation is similarly obtained by replacing ReLU with SoftPlus.
    """
    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO:
        raise NotImplementedError()

    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO
        raise NotImplementedError()

    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1)
        return shape


class MaxPool2d(DiffApproxModule, torch.nn.MaxPool2d):
    """
    The two-dimensional Max Pooling operation (see torch.nn.MaxPool2d).
    The approximation of this module is based on expressing the max operation of two arguments using ReLU:

    .. math::
    \\max(x,y) = ReLU(x-y) + y
    this function can be chained to obtain an arbitrary argument max operation from ReLU.
    The "main" approximation is then build by replacing ReLU with SiLU in the above formula.
    The "complementary" approximation is similarly obtained by replacing ReLU with SoftPlus.
    """
    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO:
        raise NotImplementedError()

    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO
        raise NotImplementedError()

    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape


class MaxPool3d(DiffApproxModule, torch.nn.MaxPool3d):
    """
    The three-dimensional Max Pooling operation (see torch.nn.MaxPool3d).

    **NOTE: currently this class does not provide approximations.
    Obtaining approximations will throw a NotImplementedError**

    The approximations of this module could be based on expressing the max operation of two arguments using ReLU:

    .. math::
    \\max(x,y) = ReLU(x-y) + y
    this function can be chained to obtain an arbitrary argument max operation from ReLU.
    The "main" approximation could then build by replacing ReLU with SiLU in the above formula.
    The "complementary" approximation can similarly obtained by replacing ReLU with SoftPlus.
    """
    def _main_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO:
        raise NotImplementedError()

    def _complementary_approx_impl(self, tightness: int) -> 'DiffModule':
        # TODO
        raise NotImplementedError()

    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1, -1)
        return shape


# create a number of classes for layers that are already differentiable everywhere

class Linear(DiffModule, torch.nn.Linear):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_features, )
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_features, )
        return shape


class Identity(ElementwiseModule, DiffModule, torch.nn.Identity):
    pass


# various convolution classes

class Conv1d(DiffModule, torch.nn.Conv1d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1)
        return shape


class Conv2d(DiffModule, torch.nn.Conv2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1, -1)
        return shape


class Conv3d(DiffModule, torch.nn.Conv2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1, -1, -1)
        return shape


# average pool (differentiable)

class AvgPool1d(DiffModule, torch.nn.AvgPool1d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1)
        return shape


class AvgPool2d(DiffModule, torch.nn.AvgPool2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1, -1)
        return shape


class AvgPool3d(DiffModule, torch.nn.AvgPool3d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.in_channels, -1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.out_channels, -1, -1, -1)
        return shape


# differentiable activation functions
class Sigmoid(ElementwiseModule, DiffModule, torch.nn.Sigmoid):
    pass


class LogSigmoid(ElementwiseModule, DiffModule, torch.nn.LogSigmoid):
    pass


class Tanh(ElementwiseModule, DiffModule, torch.nn.Tanh):
    pass


# a continuously differentiable exponential linear unit (ELU) activation function
class CELU(ElementwiseModule, DiffModule, torch.nn.CELU):
    pass


class GELU(ElementwiseModule, DiffModule, torch.nn.GELU):
    pass


class Softsign(ElementwiseModule, DiffModule, torch.nn.Softsign):
    pass


class Tanhshrink(ElementwiseModule, DiffModule, torch.nn.Tanhshrink):
    pass


class Softmin(ElementwiseModule, DiffModule, torch.nn.Softmin):
    pass


class Softmax(ElementwiseModule, DiffModule, torch.nn.Softmax):
    pass


class Softmax2d(DiffModule, torch.nn.Softmax2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape


class LogSoftmax(ElementwiseModule, DiffModule, torch.nn.LogSoftmax):
    pass


# dropout layers

class Dropout(AnyInputsShapeModule, DiffModule, torch.nn.Dropout):
    pass


class Dropout2d(DiffModule, torch.nn.Dropout2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1)
        return shape


class Dropout3d(DiffModule, torch.nn.Dropout3d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (-1, -1, -1, -1)
        return shape


class AlphaDropout(AnyInputsShapeModule, DiffModule, torch.nn.AlphaDropout):
    pass


# batch normalization layers

# the norm layers have some restrictions on the input shape,
# but there are multiple options with different dimensions, which can not be expressed with the current format
class BatchNorm1d(AnyInputsShapeModule, DiffModule, torch.nn.BatchNorm1d):
    pass


class BatchNorm2d(AnyInputsShapeModule, DiffModule, torch.nn.BatchNorm2d):
    pass


class BatchNorm3d(AnyInputsShapeModule, DiffModule, torch.nn.BatchNorm3d):
    pass


class GroupNorm(AnyInputsShapeModule, DiffModule, torch.nn.GroupNorm):
    pass


class SyncBatchNorm(AnyInputsShapeModule, DiffModule, torch.nn.SyncBatchNorm):
    pass


class InstanceNorm1d(DiffModule, torch.nn.InstanceNorm1d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1)
        return shape


class InstanceNorm2d(DiffModule, torch.nn.InstanceNorm2d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1, -1)
        return shape


class InstanceNorm3d(DiffModule, torch.nn.InstanceNorm3d):
    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1, -1, -1)
        return shape

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        shape = (self.num_features, -1, -1, -1)
        return shape


# also here the input shape is restricted, but in a way to complicated to express with the current format
class LayerNorm(AnyInputsShapeModule, DiffModule, torch.nn.LayerNorm):
    pass


class LocalResponseNorm(AnyInputsShapeModule, DiffModule, torch.nn.LocalResponseNorm):
    pass


# other layers

class Flatten(AnyInputsShapeModule, DiffModule, torch.nn.Flatten):
    pass


class Unflatten(AnyInputsShapeModule, DiffModule, torch.nn.Unflatten):
    pass
