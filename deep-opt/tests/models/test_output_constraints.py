import unittest

import torch
from torch import tensor

from deep_opt import NeuralNetwork
from deep_opt.models import diff_approx
from deep_opt.models.property import Property, MultiVarProperty, OutputDistanceConstraint
from models import BoxConstraint, ExtremumConstraint, MultiOutputExtremumConstraint, \
    SameExtremumConstraint, OutputsComparisonConstraint


class TestOutputConstraints(unittest.TestCase):

    # network used to test constraints by directly supplying the input data
    id_out2 = NeuralNetwork(modules=[diff_approx.Identity()], inputs_shape=(2,), outputs_shape=(2,))
    id_out3 = NeuralNetwork(modules=[diff_approx.Identity()], inputs_shape=(3,), outputs_shape=(3,))

    def test_same_max_constraint(self):
        output_constraint = SameExtremumConstraint('max')
        prop = MultiVarProperty(
            lower_bounds=({0: 0, 1: 0}, {0: 0, 1: 0}),
            upper_bounds=({0: 1, 1: 1}, {0: 1, 1: 1}),
            numbers_of_inputs=(2, 2),
            output_constraint=output_constraint,
            property_name='test property 5'
        )

        violated = [
            [1, 0, 0, 1], [2, 1, 0, 2], [-2, -1, 4, -4], [0, 0.5, 4, 3.9]
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [1, 0, 1, 0], [2, -1, 4, 3.9], [-2, -1, 4, 4], [0, 5, -2, -1.99],
            [0, 0, 0, 0], [1.5, 1.5, 1, -1]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out2) < 0
            assert not prop.property_satisfied(sample, self.id_out2)

        assert all(prop.satisfaction_function(violated_2, self.id_out2) < 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out2))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out2) >= 0
            assert prop.property_satisfied(sample, self.id_out2)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out2) >= 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out2))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out2).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out2) < 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out2) >= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out2)[:len(violated)] < 0)
        assert all(prop.satisfaction_function(mixed, self.id_out2)[len(violated):] >= 0)

    def test_output_distance_constraint(self):
        output_constraint = OutputDistanceConstraint(threshold=1, norm_order=1)
        prop = MultiVarProperty(
            lower_bounds=({0: -10, 1: -10}, {0: -10, 1: -10}),
            upper_bounds=({0: 10, 1: 10}, {0: 10, 1: 10}),
            numbers_of_inputs=(3, 3),
            output_constraint=output_constraint,
            property_name='test property 6'
        )

        violated = [
            [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, -2, 0], [1, 0, -1, 1, 0, 1],
            [-1, -2, 3, -1.5, -1.5, 3.25], [1, 2, 3, 3, 2, 1]
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [-1, -1, -1, 0, -1, -1],
            [-3, 2, -1, -3, 1.5, -0.5], [7, 3, -5, 7.25, 3.25, -5.5],
            [0.25, 0.125, 0.1, 0, 0, 0]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out3) < 0
            assert not prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(violated_2, self.id_out3) < 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out3))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out3) >= 0
            assert prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out3) >= 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out3))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out3).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) < 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) >= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[:len(violated)] < 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[len(violated):] >= 0)

    def test_multi_max_constraint(self):
        output_constraint = MultiOutputExtremumConstraint('strict_max', 'in', [0, 2])
        prop = Property({}, {}, output_constraint, property_name='test property 5')

        violated = [
            [-1, 0, -2], [0, 0, -2], [0, 0, 0], [-2, 0, -2]
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [-1, 0, 1], [1, 0, 1], [0, 2, 3], [3, 2, 1]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out3) <= 0
            assert not prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(violated_2, self.id_out3) <= 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out3))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out3) > 0
            assert prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out3) > 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out3))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out3).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) <= 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) > 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[:len(violated)] <= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[len(violated):] > 0)

    def test_multi_min_constraint(self):
        output_constraint = MultiOutputExtremumConstraint('min', 'not_in', [0, 2])
        prop = Property({}, {}, output_constraint, property_name='test property 4')

        violated = [
            [-1, 0, -2], [0, 0, -2], [0, 0, 0], [-2, -2, -1]
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [1, 0, 1], [0, -1, 0]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out3) <= 0
            assert not prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(violated_2, self.id_out3) <= 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out3))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out3) > 0
            assert prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out3) > 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out3))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out3).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) <= 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) > 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[:len(violated)] <= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[len(violated):] > 0)

    def test_max_constraint(self):
        output_constraint = ExtremumConstraint(1, '!=', 'max')
        prop = Property({}, {}, output_constraint, property_name='test property 3')

        violated = [
            [-1, 1, 0], [1, 1, -1], [0, 0, 0], [-1, 1, -1]
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [0, -1, 1], [0, -1, -1], [1, 0, 1], [1, 0, -1]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out3) <= 0
            assert not prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(violated_2, self.id_out3) <= 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out3))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out3) > 0
            assert prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out3) > 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out3))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out3).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) <= 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) > 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[:len(violated)] <= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[len(violated):] > 0)

    def test_min_constraint(self):
        output_constraint = ExtremumConstraint(1, '==', 'strict_min')
        prop = Property({}, {}, output_constraint, property_name='test property 2')

        violated = [
            [-1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 1, -1],
            [0, -1, -1], [1, 1, 1]  # these are violations since the minimum should be strict
        ]
        violated_1 = [tensor([x]) for x in violated]
        violated_2 = tensor(violated)
        satisfied = [
            [0, -1, 1], [0, -1, -0.9], [1, 0, 1], [1, 0.999, 1]
        ]
        satisfied_1 = [tensor([x]) for x in satisfied]
        satisfied_2 = tensor(satisfied)

        for sample in violated_1:
            assert prop.satisfaction_function(sample, self.id_out3) <= 0
            assert not prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(violated_2, self.id_out3) <= 0)
        assert all(~prop.property_satisfied(violated_2, self.id_out3))

        for sample in satisfied_1:
            assert prop.satisfaction_function(sample, self.id_out3) > 0
            assert prop.property_satisfied(sample, self.id_out3)

        assert all(prop.satisfaction_function(satisfied_2, self.id_out3) > 0)
        assert all(prop.property_satisfied(satisfied_2, self.id_out3))

        mixed = torch.vstack([violated_2, satisfied_2])
        assert prop.satisfaction_function(mixed, self.id_out3).shape == (len(violated) + len(satisfied),)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) <= 0)
        assert not all(prop.satisfaction_function(mixed, self.id_out3) > 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[:len(violated)] <= 0)
        assert all(prop.satisfaction_function(mixed, self.id_out3)[len(violated):] > 0)

    def test_outputs_comparison_constraint(self):
        output_constraint = OutputsComparisonConstraint(2, '<=', 0)
        prop = Property({}, {}, output_constraint, property_name='test property 5')

        assert prop.satisfaction_function(tensor([[0, 0, 9]]), self.id_out3) < 0
        assert prop.satisfaction_function(tensor([[1, 3, 2]]), self.id_out3) < 0
        assert prop.satisfaction_function(tensor([[-3, 0, 0]]), self.id_out3) < 0
        assert prop.satisfaction_function(tensor([[0, 1, 0]]), self.id_out3) >= 0
        assert prop.satisfaction_function(tensor([[0, 7, -9]]), self.id_out3) >= 0

        assert not prop.property_satisfied(tensor([[0, 0, 9]]), self.id_out3)
        assert not prop.property_satisfied(tensor([[1, 3, 2]]), self.id_out3)
        assert not prop.property_satisfied(tensor([[-3, 0, 0]]), self.id_out3)
        assert prop.property_satisfied(tensor([[0, 1, 0]]), self.id_out3)
        assert prop.property_satisfied(tensor([[0, 7, -9]]), self.id_out3)

        violations = prop.satisfaction_function(
            tensor([[0, 0, 9], [1, 3, 2], [-3, 0, 0]]),
            self.id_out3
        )
        assert violations.shape == (3, )
        assert all(violations < 0)
        satisfying = prop.satisfaction_function(
            tensor([[0, 1, 0], [0, 7, -9], [-9, 2, -9.1]]),
            self.id_out3
        )
        assert satisfying.shape == (3, )
        assert all(satisfying >= 0)

        mixed = prop.satisfaction_function(
            tensor([[0, 0, 9], [1, 3, 2], [-3, 0, 0], [0, 1, 0], [0, 7, -9], [-9, 2, -9.1]]),
            self.id_out3
        )
        assert mixed.shape == (6, )
        assert not all(mixed < 0)
        assert not all(mixed >= 0)

    def test_box_constraint(self):
        output_constraint = BoxConstraint(1, '>', 15)
        prop = Property({}, {}, output_constraint, property_name='test property 1')

        assert prop.satisfaction_function(tensor([[0,   0]]), self.id_out2) <= 0
        assert prop.satisfaction_function(tensor([[16,  0]]), self.id_out2) <= 0
        assert prop.satisfaction_function(tensor([[0,  15]]), self.id_out2) <= 0
        assert prop.satisfaction_function(tensor([[0,  16]]), self.id_out2) > 0
        assert prop.satisfaction_function(tensor([[16, 16]]), self.id_out2) > 0

        assert not prop.property_satisfied(tensor([[0,   0]]), self.id_out2)
        assert not prop.property_satisfied(tensor([[16,  0]]), self.id_out2)
        assert not prop.property_satisfied(tensor([[0,  15]]), self.id_out2)
        assert prop.property_satisfied(tensor([[0,  16]]), self.id_out2)
        assert prop.property_satisfied(tensor([[16, 16]]), self.id_out2)

        violations = prop.satisfaction_function(
            tensor([[0, 0], [16, 0], [0, 15], [16, 15]]),
            self.id_out2
        )
        assert violations.shape == (4, )
        assert all(violations <= 0)
        satisfying = prop.satisfaction_function(
            tensor([[0, 16], [16, 16], [0, 17]]),
            self.id_out2
        )
        assert satisfying.shape == (3, )
        assert all(satisfying > 0)

        mixed = prop.satisfaction_function(
            tensor([[0, 17], [5, -3], [14, 15], [15, 15.1], [16, 16]]),
            self.id_out2
        )
        assert mixed.shape == (5, )
        assert not all(mixed <= 0)
        assert not all(mixed > 0)


if __name__ == '__main__':
    unittest.main()
