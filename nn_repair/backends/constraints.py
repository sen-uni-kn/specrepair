from collections import defaultdict
from typing import Callable, Sequence, Tuple, Dict
from enum import Enum, auto

import torch
from deep_opt import Property, NeuralNetwork


class PenaltyFunction(Enum):
    """
    Defines the classes of penalty functions that can be used.
     - QUADRATIC: quadratic penalty (not exact): 1/2|g(x)|^2
     - L1: L1-norm penalty (exact): |g(x)|
    """
    QUADRATIC = auto()
    L1 = auto()
    L1_PLUS_QUADRATIC = auto()

    def __str__(self):
        if self == PenaltyFunction.QUADRATIC:
            return 'quadratic'
        elif self == PenaltyFunction.L1:
            return 'l1'
        elif self == PenaltyFunction.L1_PLUS_QUADRATIC:
            return 'l1 + quadratic'
        else:
            raise NotImplementedError()


class BarrierFunction(Enum):
    """
    Defines the classes of barrier functions that can be used.
     - NATURAL_LOG: natural logarithm barrier function: -ln(c(x))
     - RECIPROCAL: the reciprocal barrier function: 1/c(x)
    """
    NATURAL_LOG = auto()
    RECIPROCAL = auto()

    def __str__(self):
        if self == BarrierFunction.NATURAL_LOG:
            return 'natural logarithm'
        elif self == BarrierFunction.RECIPROCAL:
            return 'reciprocal'
        else:
            raise NotImplementedError()


class ConstraintFunctionFactory:
    """
    Instances of this class create torch compatible functions
    measuring the violation of a property for a certain network and a specific network inputs (e.g. a counterexample).
    <br>
    A constraint function is a function that is only zero iff the property is satisfied by the network for the
    specified network inputs.
    """

    def __init__(self, network: NeuralNetwork = None, prop: Property = None, satisfaction_eps: float = 0):
        self._network = network
        self._property = prop
        self._satisfaction_eps = satisfaction_eps

    def with_network(self, network: NeuralNetwork) -> 'ConstraintFunctionFactory':
        """
        Sets the network for which this factory creates create violation functions.<br>
        The network may be updated during the lifetime of the factory and will then replace
        the previous network.
        :return: the very same object as this method was called on, for call chaining.
        """
        self._network = network
        return self

    def with_property(self, prop: Property) -> 'ConstraintFunctionFactory':
        """
        Sets the property for which this factory creates violation functions.<br>
        The property may be updated during the lifetime of the factory and will then replace
        the previous property.

        The property needs to have it's satisfaction/violation boundary at 0.
        This means that any value of the properties satisfaction_function that is > 0 needs to indicate
        satisfaction.
        :param prop: The property to use
        :return: the very same object as this method was called on, for call chaining.
        """
        self._property = prop
        return self

    def with_satisfaction_eps(self, satisfaction_eps: float) -> 'ConstraintFunctionFactory':
        """
        Sets the satisfaction epsilon, which is used to shift the actual constant of an inequality constraint
        by a bit to guarantee feasibility if feasible points are approached from the infeasible domain.
        Any inequality constraint g(x) >= 0 will be transformed to g(x) >= eps where eps is the satisfaction_eps
        set with this method.
        :param satisfaction_eps: The small constant that will be used to shift inequality constraints generated
        from the specification
        :return: the very same object as this method was called on, for call chaining.
        """
        self._satisfaction_eps = satisfaction_eps
        return self

    def clone(self) -> 'ConstraintFunctionFactory':
        return ConstraintFunctionFactory(self._network, self._property, self._satisfaction_eps)

    def create_constraint_function(self, inputs: torch.Tensor, batched_already=False) -> Callable[[], torch.Tensor]:
        """
        Creates a constraint function for the given network inputs and the previously specified property and network.
        The constraint function is the function that needs to be >= 0 in a constrained optimisation problem.
        The constraint function is simply the satisfaction function of the property
        shifted by the satisfaction epsilon:

            property.satisfaction_function - satisfaction_eps

        :param inputs: The network inputs for which the returned function gives the constraint
         (e.g. the counterexample).
        :param batched_already: Whether the `inputs` already has a batch dimension.
          When this is not the case, this method creates a batch dimension.
          The output is batched in any case.
        :return: A torch compatible function that is >= 0 exactly if the previously set network fulfils
         the previously set property for the given network inputs.
         The satisfaction function returns a vector with as many elements as the batch dimension of `inputs`.
         If `batched_already` is False, the output will be a vector with one element.
        """
        network = self._network
        prop = self._property
        input_tensor = inputs.detach().clone()
        if not batched_already:
            input_tensor.unsqueeze_(0)

        def constraint() -> torch.Tensor:
            property_func = prop.satisfaction_function(input_tensor, network)
            return property_func - self._satisfaction_eps
        return constraint

    def create_vector_constraint_function(self, inputs_with_properties: Dict[Property, Sequence[torch.Tensor]]) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a constraint function that contains constraint values for multiple inputs, potentially
        for different properties, for the previously specified network.

        The constraint function is the function that needs to be >= 0 in a constrained optimisation problem.
        The constraint function is simply the satisfaction function of the property
        shifted by the satisfaction epsilon:

            property.satisfaction_function - satisfaction_eps

        To obtain a vector function, these constraint functions are simply stacked.
        This method is the foundation for many further vectorized functions.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         gives the constraints.
         Each property is mapped to its network inputs.
        :return: A torch compatible vector function, in which each element is >= 0 if the previously set
         network fulfils the corresponding property for the corresponding network inputs.
        """
        initial_property = self._property
        constraint_functions = []
        for prop, inputs_sequence in inputs_with_properties.items():
            self.with_property(prop)
            inputs_sequence = tuple(inputs_sequence)
            if len(inputs_sequence) > 0:
                inputs = torch.vstack(inputs_sequence)
                constraint_functions.append(self.create_constraint_function(inputs, batched_already=True))
        self.with_property(initial_property)

        def stacked() -> torch.Tensor:
            # the return value of c is a 1-d tensor. Stack along this single dimension, do not create a new dimension
            return torch.hstack([c() for c in constraint_functions])

        return stacked

    def create_quadratic_penalty(self, inputs: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Creates a violation function for the given network inputs and the previously specified property and network.
        This function creates a quadratic penalty:

        .. math::
            \\frac{1}{2} * \\min(property.satisfaction\\_function - satisfaction\\_eps, 0)^2

        :param inputs: The network inputs for which the returned function will measure property violation.
        :return: A torch compatible function that is zero exactly if the previously set network fulfils
         the previously set property for the given network inputs.
        """
        constraint = self.create_constraint_function(inputs)
        return self._quadratic_penalty_impl(constraint)

    def create_vector_quadratic_penalty(self, inputs_with_properties: Dict[Property, Sequence[torch.Tensor]]) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a function returning a vector of quadratic penalty functions.
        This vector is simply a stacked version of the quadratic penalty functions for the given network inputs.
        However some operations are applied in bulk for the whole tensor, making this version faster.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        constraint = self.create_vector_constraint_function(inputs_with_properties)
        return self._quadratic_penalty_impl(constraint)

    @staticmethod
    def _quadratic_penalty_impl(constraint) -> Callable[[], torch.Tensor]:
        # this works for both vector and scalar constraint functions
        def violation() -> torch.Tensor:
            return 0.5 * torch.square(torch.minimum(constraint(), torch.zeros(())))
        return violation

    def create_l1_penalty(self, inputs: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Creates a violation function for the given network inputs and the previously specified property and network.
        This function creates a :math:`\\ell_1` penalty:

        .. math::
            -\\min(property.satisfaction\\_function - satisfaction\\_eps, 0)

        :param inputs: The network inputs for which the returned function will measure property violation.
        :return: A torch compatible function that is zero exactly if the previously set network fulfils
         the previously set property for the given network inputs.
        """
        constraint = self.create_constraint_function(inputs)
        return self._l1_penalty_impl(constraint)

    def create_vector_l1_penalty(self, inputs_with_properties: Dict[Property, Sequence[torch.Tensor]]) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a function returning a vector of l1 penalty functions.
        This vector is simply a stacked version of the l1 penalty functions for the given network inputs.
        However, some operations are applied in bulk for the whole tensor, making this version faster.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        constraint = self.create_vector_constraint_function(inputs_with_properties)
        return self._l1_penalty_impl(constraint)

    @staticmethod
    def _l1_penalty_impl(constraint) -> Callable[[], torch.Tensor]:
        # this works for both vector and scalar constraint functions
        def violation() -> torch.Tensor:
            return - torch.minimum(constraint(), torch.zeros(()))
        return violation

    def create_l1_plus_quadratic_penalty(self, inputs: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Creates a violation function for the given network inputs and the previously specified property and network.
        This function the sum of a :math:`\\ell_1` penalty and a quadratic penalty
        (similar to the augmented lagrangian):

        .. math::
            -\\min(property.satisfaction\\_function - satisfaction\\_eps, 0)
            + \\frac{1}{2}(\\min(property.satisfaction\\_function - satisfaction\\_eps, 0))^2

        :param inputs: The network inputs for which the returned function will measure property violation.
        :return: A torch compatible function that is zero exactly if the previously set network fulfils
         the previously set property for the given network inputs.
        """
        constraint = self.create_constraint_function(inputs)
        return self._l1_plus_quadratic_penalty_impl(constraint)

    def create_vector_l1_plus_quadratic_penalty(self, inputs_with_properties: Dict[Property, Sequence[torch.Tensor]]) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a function returning a vector of combined l1 and quadratic penalty functions.
        This vector is simply a stacked version of the l1 plus quadratic penalty functions for the given network inputs.
        However some operations are applied in bulk for the whole tensor, making this version faster.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        constraint = self.create_vector_constraint_function(inputs_with_properties)
        return self._l1_plus_quadratic_penalty_impl(constraint)

    @staticmethod
    def _l1_plus_quadratic_penalty_impl(constraint) -> Callable[[], torch.Tensor]:
        # this works for both vector and scalar constraint functions
        def violation() -> torch.Tensor:
            violation_only = torch.minimum(constraint(), torch.zeros(()))
            return - violation_only + 0.5 * torch.square(violation_only)
        return violation

    def create_penalty_function(self, inputs: torch.Tensor, penalty_type: PenaltyFunction) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a penalty function for the given network inputs and the previously specified property and network.

        :param inputs: The network inputs for which the returned function will measure property violation.
        :param penalty_type: The barrier_type of penalty function to create.
        :return: A torch compatible function that is zero exactly if the previously set network fulfils
         the previously set property for the given network inputs.
        """
        if penalty_type == PenaltyFunction.QUADRATIC:
            return self.create_quadratic_penalty(inputs)
        elif penalty_type == PenaltyFunction.L1:
            return self.create_l1_penalty(inputs)
        elif penalty_type == PenaltyFunction.L1_PLUS_QUADRATIC:
            return self.create_l1_plus_quadratic_penalty(inputs)
        else:
            raise ValueError(f"Unknown penalty function barrier_type: {penalty_type}")

    def create_vector_penalty_function(self, inputs_with_properties: Dict[Property, Sequence[torch.Tensor]],
                                       penalty_type: PenaltyFunction) -> Callable[[], torch.Tensor]:
        """
        Creates penalty functions in their vectorized versions.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :param penalty_type: The barrier_type of penalty function to create.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        if penalty_type == PenaltyFunction.QUADRATIC:
            return self.create_vector_quadratic_penalty(inputs_with_properties)
        elif penalty_type == PenaltyFunction.L1:
            return self.create_vector_l1_penalty(inputs_with_properties)
        elif penalty_type == PenaltyFunction.L1_PLUS_QUADRATIC:
            return self.create_vector_l1_plus_quadratic_penalty(inputs_with_properties)
        else:
            raise ValueError(f"Unknown penalty function barrier_type: {penalty_type}")

    def create_ln_barrier(self, inputs: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Creates a barrier function for the given network inputs and the previously specified property and network.
        This function creates a logarithmic barrier using the natural logarithm:
            - ln(property.satisfaction_function)
        This barrier function is unbounded as property.satisfaction_function -> infinity
        Note that satisfaction_eps is not used, since this barrier function is undefined for 0 in any case.

        :param inputs: The network inputs for which the barrier function should be defined.
        :return: A torch compatible function that has a function value < infinity if the previously set network
         fulfils the previously set property for the given network inputs.
         When the property is violated the barrier function has a value of + infinity.
         The larger the satisfaction function of the property, the smaller the barrier function is.
        """
        network = self._network
        prop = self._property
        input_tensor = inputs.clone().detach().unsqueeze(0)

        def ln_barrier() -> torch.Tensor:
            property_func = prop.satisfaction_function(input_tensor, network)
            return torch.where(
                torch.gt(property_func, 0),
                -torch.log(property_func),
                torch.tensor(float('inf'))
            )
        return ln_barrier

    def create_vector_ln_barrier(self, inputs_with_properties: Sequence[Tuple[Property, torch.Tensor]]):
        """
        Creates a function returning a vector of ln barrier functions.
        This vector is simply a stacked version of the ln barrier functions for the given network inputs.
        However some operations are applied in bulk for the whole tensor, making this version faster.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        network = self._network
        inputs_with_properties = [(p, t.clone().detach().unsqueeze(0)) for p, t in inputs_with_properties]

        def ln_barrier() -> torch.Tensor:
            # prop.satisfaction_function is 1-d tensor. Stack along this existing dimension, so not create a new one
            stacked_satisfaction_functions = torch.vstack([
                prop.satisfaction_function(inputs, network) for prop, inputs in inputs_with_properties
            ])
            return torch.where(
                torch.gt(stacked_satisfaction_functions, 0),
                -torch.log(stacked_satisfaction_functions),
                torch.tensor(float('inf'))
            )
        return ln_barrier

    def create_reciprocal_barrier(self, inputs: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Creates a barrier function for the given network inputs and the previously specified property and network.
        This function creates a reciprocal barrier with function values set to + infinity for x < 0:
            1 / property.satisfaction_function
        This barrier function is bounded by 0 as property.satisfaction_function -> infinity
        Note that satisfaction_eps is not used, since this barrier function is undefined for 0 in any case.
        :param inputs: The network inputs for which the barrier function should be defined.
        :return: A torch compatible function that has a function value < infinity if the previously set network
        fulfils the previously set property for the given network inputs.
        When the property is violated the barrier function has a value of + infinity.
        The larger the satisfaction function of the property, the smaller the barrier function is.
        """
        network = self._network
        prop = self._property
        input_tensor = inputs.clone().detach().unsqueeze(0)

        def reciprocal_barrier() -> torch.Tensor:
            network_output = network(input_tensor)
            property_func: torch.Tensor = prop.satisfaction_function(input_tensor, network)
            return torch.where(
                torch.gt(property_func, 0),
                1 / property_func,
                torch.tensor(float('inf'))
            )
        return reciprocal_barrier

    def create_vector_reciprocal_barrier(self, inputs_with_properties: Sequence[Tuple[Property, torch.Tensor]]):
        """
        Creates a function returning a vector of reciprocal barrier functions.
        This vector is simply a stacked version of the reciprocal barrier functions for the given network inputs.
        However some operations are applied in bulk for the whole tensor, making this version faster.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :return: A torch compatible vector function whose elements are zero exactly if the previously set network
         fulfils the corresponding property for the corresponding network inputs.
        """
        network = self._network
        inputs_with_properties = [(p, t.clone().detach().unsqueeze(0)) for p, t in inputs_with_properties]

        def reciprocal_barrier() -> torch.Tensor:
            # prop.satisfaction_function is 1-d tensor. Stack along this existing dimension, so not create a new one
            stacked_satisfaction_functions = torch.vstack([
                prop.satisfaction_function(inputs, network) for prop, inputs in inputs_with_properties
            ])
            return torch.where(
                torch.gt(stacked_satisfaction_functions, 0),
                1 / stacked_satisfaction_functions,
                torch.tensor(float('inf'))
            )
        return reciprocal_barrier

    def create_barrier_function(self, inputs: torch.Tensor, barrier_type: BarrierFunction) \
            -> Callable[[], torch.Tensor]:
        """
        Creates a barrier function for the given network input and the previously specified property and network.

        Note that satisfaction_eps may not be used in some cases.
         - BarrierFunction.NATURAL_LOG does not use satisfaction_eps
         - BarrierFunction.RECIPROCAL does not use satisfaction_eps
        See the individual functions for details.

        :param inputs: The network inputs for which the barrier function should be defined.
        :param barrier_type: The type of barrier function to create.
        :return: A torch compatible function that has small(er) function value if the previously set network
         fulfils the previously set property for the given network input.
         When the property is violated the barrier function is large or even has a value of + infinity.
         The larger the satisfaction function of the property, the smaller the barrier function is.
        """
        if barrier_type == BarrierFunction.NATURAL_LOG:
            return self.create_ln_barrier(inputs)
        elif barrier_type == BarrierFunction.RECIPROCAL:
            return self.create_reciprocal_barrier(inputs)
        else:
            raise ValueError(f"Unknown barrier function barrier_type: {barrier_type}")

    def create_vector_barrier_function(self, inputs_with_properties: Sequence[Tuple[Property, torch.Tensor]],
                                       barrier_type: BarrierFunction) -> Callable[[], torch.Tensor]:
        """
        Creates barrier functions in their vectorized versions.

        :param inputs_with_properties: The network inputs and corresponding properties for which the returned function
         measures property violation.
        :param barrier_type: The type of barrier function to create.
        :return: A torch compatible vector function whose elements are small(er) if the previously set network
         fulfils the corresponding property for the corresponding network input.
         When a property is violated for the corresponding counterexample, the corresponding element of the
         barrier function is large or even has a value of + infinity.
         The larger the satisfaction function of the property, the smaller the corresponding element of the
         barrier function is.
        """
        if barrier_type == BarrierFunction.NATURAL_LOG:
            return self.create_vector_ln_barrier(inputs_with_properties)
        elif barrier_type == BarrierFunction.RECIPROCAL:
            return self.create_vector_reciprocal_barrier(inputs_with_properties)
        else:
            raise ValueError(f"Unknown barrier function barrier_type: {barrier_type}")
