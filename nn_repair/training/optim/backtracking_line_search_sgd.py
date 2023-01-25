import torch
from torch.optim import Optimizer

from nn_repair.training.termination import TerminationCriterion


class BacktrackingLineSearchSGD(Optimizer):
    """
    Implements stochastic gradient descent with a simple line search algorithm: backtracking line search.

    This optimizers ``step`` method requires a closure called.
    """

    def __init__(self, params, start_lr=1, lr_reduce_step=0.5, armijo_goldstein_constant=0.5, max_steps=100):
        assert start_lr > 0
        assert 0 < lr_reduce_step < 1
        assert 0 < armijo_goldstein_constant < 1

        super().__init__(params, {})
        self.state['lr'] = start_lr
        self.state['tau'] = lr_reduce_step
        self.state['c'] = armijo_goldstein_constant
        self.state['max_steps'] = max_steps
        self.state['last_steps'] = -1

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            initial_loss = closure()

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
        flat_grad = torch.hstack([g.flatten() for g in grads])
        m = torch.dot(flat_grad, flat_grad)
        c = self.state['c']

        lr = self.state['lr']
        # perform the first step with the initial learning rate
        for p, g in zip(params_with_grad, grads):
            p.data -= lr * g

        tau = self.state['tau']
        step = 0
        for step in range(self.state['max_steps']):
            if initial_loss - closure() >= lr * c * m:
                break
            new_lr = tau * lr
            # we have decreased the learning rate
            # we now have to walk back the difference between the old and the new learning rate
            # to simulate performing a step with the new learning rate from the initial parameter values
            for p, g in zip(params_with_grad, grads):
                p.data += (lr - new_lr) * g
            lr = new_lr
        self.state['last_steps'] = step

        return initial_loss


class BacktrackingLineSearchMaxStepsPerformed(TerminationCriterion):
    """
    Terminates training of the BacktrackingLineSearchSGD optimizer reached it's
    maximum number of iterations in the last optimizer step.
    """

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        optimizer = training_loop.optimizer
        if isinstance(optimizer, BacktrackingLineSearchSGD):
            return optimizer.state['last_steps'] >= optimizer.state['max_steps'] - 1
