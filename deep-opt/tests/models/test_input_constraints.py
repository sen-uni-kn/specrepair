import itertools
import unittest

import torch
from torch import tensor
import numpy as np

from deep_opt.models import DistanceConstraint

class TestInputConstraints(unittest.TestCase):

    def test_distance_constraint_creation(self):
        DistanceConstraint(threshold=1, norm_order='inf')
        DistanceConstraint(threshold=1, norm_order=float('inf'))
        DistanceConstraint(threshold=1, norm_order=np.inf)

    def test_l_inf_distance_constraint(self):
        input_constraint = DistanceConstraint(threshold=1, norm_order='inf')

        violated = [
            ([0, 0, 0], [2, 0, 0]),
            ([1.1, 0, 0], [0, 0, 0]),
            ([1, 1, -2], [0, 1, 0]),
            ([-1, 0, 2], [0.5, 1, 0]),
            ([-1, -2, -0.5], [-0.5, -0.5, -0.5]),
            ([-1, -2, -0.5], [-3, -1, -0.5]),
            ([-1, -2, -0.5], [-0.5, -1, -4]),
            ([-7, -5, -3], [7, 5, 3]),
        ]
        satisfied = [
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [1, 0, 0]),
            ([1, 1, 1], [1, 0.5, 0]),
            ([0.5, -1, -4], [-0.25, -1.1, -3.2]),
            ([-7, -5, -3], [-8, -6, -4]),
            ([-7, -5, -3], [-6, -5, -3]),
        ]
        violated_1 = [(tensor([x], dtype=torch.float32), tensor([y], dtype=torch.float32)) for x, y in violated]
        violated_2 = (tensor([x for x, _ in violated]), tensor([y for _, y in violated]))
        satisfied_1 = [(tensor([x], dtype=torch.float32), tensor([y], dtype=torch.float32)) for x, y in satisfied]
        satisfied_2 = (tensor([x for x, _ in satisfied]), tensor([y for _, y in satisfied]))

        for x, y in violated_1:
            assert input_constraint.satisfaction_function((x, y)) < 0
            assert input_constraint.satisfaction_function((y, x)) < 0
            assert input_constraint.satisfaction_function((x, x)) >= 0
            assert input_constraint.satisfaction_function((y, y)) >= 0
            assert input_constraint.satisfaction_function((x+0.1, x)) >= 0
            assert input_constraint.satisfaction_function((y, y+0.99)) >= 0

        assert all(input_constraint.satisfaction_function(violated_2) < 0)
        assert all(input_constraint.satisfaction_function((violated_2[1], violated_2[0])) < 0)
        assert all(input_constraint.satisfaction_function((violated_2[0], violated_2[0])) >= 0)
        assert all(input_constraint.satisfaction_function((violated_2[1], violated_2[1])) >= 0)
        assert all(input_constraint.satisfaction_function((violated_2[0], violated_2[0]+0.25)) >= 0)
        assert all(input_constraint.satisfaction_function((violated_2[1]-0.7, violated_2[1])) >= 0)

        for x, y in satisfied_1:
            assert input_constraint.satisfaction_function((x, y)) >= 0
            assert input_constraint.satisfaction_function((y, x)) >= 0
            assert input_constraint.satisfaction_function((x, x)) >= 0
            assert input_constraint.satisfaction_function((y, y)) >= 0
            assert input_constraint.satisfaction_function((x+0.1, x)) >= 0
            assert input_constraint.satisfaction_function((y, y+0.99)) >= 0

        assert all(input_constraint.satisfaction_function(satisfied_2) >= 0)
        assert all(input_constraint.satisfaction_function((satisfied_2[1], satisfied_2[0])) >= 0)
        assert all(input_constraint.satisfaction_function((satisfied_2[0], satisfied_2[0])) >= 0)
        assert all(input_constraint.satisfaction_function((satisfied_2[1], satisfied_2[1])) >= 0)
        assert all(input_constraint.satisfaction_function((satisfied_2[0]-0.1, satisfied_2[0]+0.2)) >= 0)
        assert all(input_constraint.satisfaction_function((satisfied_2[1]-0.2, satisfied_2[1]-0.3)) >= 0)

        mixed = (
            torch.vstack([violated_2[0], satisfied_2[0]]),
            torch.vstack([violated_2[1], satisfied_2[1]]),
        )
        assert input_constraint.satisfaction_function(mixed).shape == (len(violated) + len(satisfied), )
        assert not all(input_constraint.satisfaction_function(mixed) < 0)
        assert not all(input_constraint.satisfaction_function(mixed) >= 0)
        assert all(input_constraint.satisfaction_function((mixed[0], mixed[0])) >= 0)
        assert all(input_constraint.satisfaction_function((mixed[1], mixed[1])) >= 0)
        assert all(input_constraint.satisfaction_function((mixed[0]-5, mixed[0]-5.5)) >= 0)
        assert all(input_constraint.satisfaction_function((mixed[1], mixed[1]-1)) >= 0)
        assert all(input_constraint.satisfaction_function(mixed)[:len(violated)] < 0)
        assert all(input_constraint.satisfaction_function(mixed)[len(violated):] >= 0)

    def test_l_inf_distance_constraint_projection(self):
        input_constraint = DistanceConstraint(threshold=1, norm_order='inf')

        violated = [
            ([0, 0, 0], [2, 0, 0]),
            ([1.1, 0, 0], [0, 0, 0]),
            ([1, 1, -2], [0, 1, 0]),
            ([-1, 0, 2], [0.5, 1, 0]),
            ([-1, -2, -0.5], [-0.5, -0.5, -0.5]),
            ([-1, -2, -0.5], [-3, -1, -0.5]),
            ([-1, -2, -0.5], [-0.5, -1, -4]),
            ([-7, -5, -3], [7, 5, 3]),
        ]
        satisfied = [
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [1, 0, 0]),
            ([1, 1, 1], [1, 0.5, 0]),
            ([0.5, -1, -4], [-0.25, -1.1, -3.2]),
            ([-7, -5, -3], [-8, -6, -4]),
            ([-7, -5, -3], [-6, -5, -3]),
        ]
        violated_1 = [(tensor([x], dtype=torch.float32), tensor([y], dtype=torch.float32)) for x, y in violated]
        violated_2 = (tensor([x for x, _ in violated]), tensor([y for _, y in violated]))
        satisfied_1 = [(tensor([x], dtype=torch.float32), tensor([y], dtype=torch.float32)) for x, y in satisfied]
        satisfied_2 = (tensor([x for x, _ in satisfied]), tensor([y for _, y in satisfied]))

        for x, y in violated_1:
            projected = input_constraint.compute_projection((x, y))
            assert input_constraint.satisfaction_function(projected) >= 0
            projected = input_constraint.compute_projection((y, x))
            assert input_constraint.satisfaction_function(projected) >= 0

        assert all(input_constraint.satisfaction_function(input_constraint.compute_projection(violated_2)) >= 0)

        for x, y in satisfied_1:
            assert input_constraint.compute_projection((x, y)) == (x, y)
            assert input_constraint.compute_projection((y, x)) == (y, x)

        for x, y in itertools.chain(violated_1, satisfied_1):
            assert input_constraint.compute_projection((x, x)) == (x, x)
            assert input_constraint.compute_projection((y, y)) == (y, y)

        assert input_constraint.compute_projection(satisfied_2) == satisfied_2


if __name__ == '__main__':
    unittest.main()
