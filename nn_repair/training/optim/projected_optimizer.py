from typing import Iterable, Optional, Callable

import torch
from torch.optim import Optimizer


class ProjectedOptimizer(Optimizer):
    """
    An optimizer that projects the steps of another optimizer to certain lower and upper bounds.
    """

    def __init__(self, optimizer: Optimizer, params: Iterable[dict]):
        """

        :param optimizer: The base optimizer which steps will be projected.
        :param params: The parameters that this optimizer will project.
        Pass as a dictionary with the keys 'params', 'lower_bounds', 'upper_bounds' and 'projection_function'.
        'lower_bounds' and 'upper_bounds' may be missing, in which case no lower or upper bounds
        will be registered.
        Similarly, 'projection_function' may also be missing.
        In that case, only projection to the bounds will take place.
        """
        defaults = {
            'lower_bounds': torch.tensor(-float('inf')),
            'upper_bounds': torch.tensor(float('inf')),
            'projection_function': None
        }
        super().__init__(params, defaults)
        self.optimizer = optimizer

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        self.optimizer.step(closure=closure)

        for group in self.param_groups:
            lower_bounds = group.get('lower_bounds')
            upper_bounds = group.get('upper_bounds')
            projection = group.get('projection_function')
            lower_bounds = [lower_bounds] if isinstance(lower_bounds, torch.Tensor) else lower_bounds
            upper_bounds = [upper_bounds] if isinstance(upper_bounds, torch.Tensor) else upper_bounds
            for param, lb, ub in zip(group['params'], lower_bounds, upper_bounds):
                param.data = projection(param)
                param.data = torch.where(torch.lt(param, lb), lb, param)
                param.data = torch.where(torch.gt(param, ub), ub, param)

        if closure is None:
            return None
        else:
            return closure()
