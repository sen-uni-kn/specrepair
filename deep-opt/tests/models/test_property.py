import unittest

import sys

import numpy as np
from random import randint

import torch
from torch import tensor

from deep_opt import NeuralNetwork
from deep_opt.models import diff_approx
from deep_opt.models.property import Property, MultiVarProperty, \
    RobustnessPropertyFactory, \
    dump_specification, load_specification, property_yaml_context_manager as yaml_cm, DistanceConstraint, \
    OutputDistanceConstraint
from deep_opt.models.property import BoxConstraint, ExtremumConstraint, MultiOutputExtremumConstraint, \
    SameExtremumConstraint, ConstraintOr, ConstraintAnd


class TestProperty(unittest.TestCase):

    # network used to test constraints by directly supplying the input data
    id_out2 = NeuralNetwork(modules=[diff_approx.Identity()], inputs_shape=(2,), outputs_shape=(2,))
    id_out3 = NeuralNetwork(modules=[diff_approx.Identity()], inputs_shape=(3,), outputs_shape=(3,))

    def test_yaml_dump(self):
        prop = Property(
            lower_bounds={0: -3, 1: 0.25, 2: -2},
            upper_bounds={0: 3, 1: 5.74},
            output_constraint=BoxConstraint(0, '>', 1.5),
            property_name='test_property'
        )
        yaml_cm.dump(prop, sys.stdout)

        prop2 = Property(
            lower_bounds={0: -3, 1: 0.25, 2: -2},
            upper_bounds={0: 3, 1: 5.74},
            output_constraint=ExtremumConstraint(2, '!=', 'min'),
            property_name='test_property2'
        )
        yaml_cm.dump(prop2, sys.stdout)

        # Note: this property is not fulfilled if output 1 and 2 are equal
        prop3 = Property(
            lower_bounds={0: -3, 1: 0, 2: -2},
            upper_bounds={0: 3, 1: 5},
            output_constraint=ConstraintOr(
                ExtremumConstraint(2, '==', 'strict_min'),
                ExtremumConstraint(1, '==', 'strict_min'),
            ),
            property_name='test_property3'
        )
        yaml_cm.dump(prop3, sys.stdout)

        prop4 = Property(
            lower_bounds={},
            upper_bounds={0: -3, 1: -2},
            output_constraint=ConstraintAnd(
                MultiOutputExtremumConstraint('strict_max', 'in', [0, 1]),
                BoxConstraint(2, '<=', 10)
            ),
            property_name='test_property4'
        )
        yaml_cm.dump(prop4, sys.stdout)

        prop5 = MultiVarProperty(
            lower_bounds=({0: 0, 1: 0}, {1: 5}),
            upper_bounds=({0: 1}, {0: 0.5, 1: 100}),
            numbers_of_inputs=(3, 2),
            output_constraint=SameExtremumConstraint('max'),
            property_name='test_property5'
        )
        yaml_cm.dump(prop5, sys.stdout)

        prop6 = MultiVarProperty(
            lower_bounds=({}, {0: 0.5, 1: 0.77}),
            upper_bounds=({}, {0: 0.6, 1: 0.66}),
            numbers_of_inputs=(5, 5),
            output_constraint=OutputDistanceConstraint(0.001, norm_order=2),
            input_constraint=DistanceConstraint(0.01),
        )
        yaml_cm.dump(prop6, sys.stdout)

    def test_yaml_load(self):
        representation = """
        !Property
        name: test_property
        input_lower_bounds: 
            0: -3
            1:  0
            2: -2
        input_upper_bounds: {0: 3, 1: 5}
        output_constraint: !BoxConstraint out0 > 1.5
        """
        prop = yaml_cm.load(representation)
        print(prop.__dict__)

        representation2 = """
        !Property
        name: test_property2
        input_lower_bounds: 
            0: 12
            1: -7
            2: 99
        input_upper_bounds: 
            0: 44
            1: -2
            2: 99.54
        output_constraint: !OutputsComparisonConstraint out2 <= out0
        """
        prop2 = yaml_cm.load(representation2)
        print(prop2.__dict__)

        representation3 = """
        !Property
        name: test_property3
        input_lower_bounds: {0: -3.5344, 1: 0, 2: -inf}
        input_upper_bounds:
            0: 0.54222
            2: inf
        output_constraint: !ExtremumConstraint out2   == strict_max(out)
        """
        prop3 = yaml_cm.load(representation3)
        print(prop3.__dict__)

        representation4 = """
        !Property
        name: test_property4
        input_lower_bounds: {0: -12, 2: 3}
        input_upper_bounds: {0: 0, 1: 5}
        output_constraint: !ConstraintAnd
            - !ExtremumConstraint out0 != min(out)
            - !BoxConstraint      out0  < 0.33
        """
        prop4 = yaml_cm.load(representation4)
        print(prop4.__dict__)

        representation5 = """
        !Property
        name: test_property5
        input_lower_bounds: {0: -1000.5, 1: -1000.5}
        input_upper_bounds: {}
        output_constraint: !ConstraintOr
            - !MultiOutputExtremumConstraint min(out) not_in {out4, out5}
            - !ConstraintAnd
                - !BoxConstraint             out4  < 0.1
                - !BoxConstraint             out5  < 0.1
        """
        prop5 = yaml_cm.load(representation5)
        print(prop5.__dict__)

        representation6 = """
        !MultiVarProperty
        name: test_property6
        input_lower_bounds: [{0: -1000.5, 1: -1000.5}, {}, {4: -2.5}]
        input_upper_bounds: [{}, {}, {4: 2.5, 7: 9}]
        numbers_of_inputs: [7, 3, 8]
        output_constraint: !ConstraintAnd
            - !SameExtremumConstraint min(out) == min(out')
            - !SameExtremumConstraint max(out) == max(out')
        """
        prop6 = yaml_cm.load(representation6)
        print(prop6.__dict__)

        representation7 = """
        !MultiVarProperty
        name: ''
        input_lower_bounds: [{}, {}]
        input_upper_bounds: [{}, {}]
        input_constraint: !DistanceConstraint L_inf distance <= 1.752
        numbers_of_inputs: [5, 5]
        output_constraint: !BoxConstraint out0 > 0.5
        """
        prop7 = yaml_cm.load(representation7)
        print(prop7.__dict__)


        representation8 = """
        !MultiVarProperty
        name: ''
        input_lower_bounds: [{}, {}]
        input_upper_bounds: [{}, {}]
        input_constraint: !DistanceConstraint L_inf distance <= 1.752
        numbers_of_inputs: [5, 5]
        output_constraint: !OutputDistanceConstraint L_1.0 distance <= 0.0033
        """
        prop8 = yaml_cm.load(representation8)
        print(prop8.__dict__)

    def test_specification_loading(self):
        with open("../resources/test_specification.yaml", 'r') as file:
            spec = load_specification(file)
        print('\n'.join(p.__dict__.__str__() for p in spec))
        print()

        dump_specification(spec, stream=sys.stdout)

    def test_specification_loading2(self):
        with open("../resources/test_specification_2.yaml", 'r') as file:
            spec = load_specification(file)
        print('\n'.join(p.__dict__.__str__() for p in spec))
        print()

        dump_specification(spec, stream=sys.stdout)

    def test_multi_var_bounds_unfolding(self):
        net = NeuralNetwork(
            mins=(0, -float('inf'), -2, -float('inf'), 4),
            maxes=(1, float('inf'), float('inf'), 7, 5),
            modules=[diff_approx.Linear(5, 5)]
        )
        prop = MultiVarProperty(
            lower_bounds=({0: -1, 1: -1, 3: 0, 4: 3}, {1: 1, 2: 0, 3: 4, 4: 5}),
            upper_bounds=({0: 2, 1: 3, 2: 4, 3: 5, 4: 6}, {0: 0.5, 2: 100, 4: 7}),
            numbers_of_inputs=(5, 5),
            output_constraint=SameExtremumConstraint('max'),
            property_name='test_property5'
        )
        bounds = prop.input_bounds(net)
        desired = (  # stacked: 10 inputs
            (0, 1), (-1, 3), (-2, 4), (0, 5), (4, 5),
            (0, 0.5), (1, float('inf')), (0, 100), (4, 7), (5, 5),
        )
        assert bounds == desired

    def test_valid_input_property(self):
        prop = Property(
            lower_bounds={0: 0.5, 1: 0, 2: -2},
            upper_bounds={0: 1.5, 1: 1, 2: 0},
            output_constraint=BoxConstraint(3, '>=', 99),
            property_name='test_property_6'
        )
        assert prop.valid_input(tensor([[1, 0.5, -1]]))
        assert prop.valid_input(tensor([[0.5, 0, -2]]))
        assert prop.valid_input(tensor([[1.5, 1, 0]]))
        assert not prop.valid_input(tensor([[1.6, 1, 0]]))
        assert not prop.valid_input(tensor([[1, -7, -1]]))
        assert not prop.valid_input(tensor([[1, 0.5, 0.000000000000000000000000000000001]]))
        batched = prop.valid_input(tensor([
            [0.75, 0, 0],
            [1.3, 1.1, -1],
            [1.0, 0.99999, -2]
        ]))
        assert batched.shape == (3,)
        assert batched[0]
        assert not batched[1]
        assert batched[2]

    def test_valid_input_multi_property(self):
        prop = MultiVarProperty(
            lower_bounds=({0: -1, 1: -1, 2: -1, 3: -1}, {0: -1, 1: -1, 2: -1, 3: -1}),
            upper_bounds=({0: 1, 1: 1, 2: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
            numbers_of_inputs=None,
            input_constraint=DistanceConstraint(0.5),
            output_constraint=SameExtremumConstraint('max')
        )
        assert prop.valid_input(tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float))
        assert prop.valid_input(tensor([[0, 0, 0, 0, -0.1, 0.1, -0.1, 0.1]]))
        assert prop.valid_input(tensor([[0.5, 0.5, 0.5, -0.5, 0, 0, 0, 0]]))
        assert prop.valid_input(tensor([[0.5, 0, 0, -0.5, 0, 0.5, -0.5, 0]]))
        assert prop.valid_input(tensor([[0.25, -0.25, 0.75, 0, 0, 0, 0.25, -0.43]]))
        assert not prop.valid_input(tensor([[0, 0, 0, -2, 0, 0, 0, -2.1]]))
        assert not prop.valid_input(tensor([[0, 0.25, 0, 0.5, 10, 0, 0, 0]]))
        assert not prop.valid_input(tensor([[0, 0, 0, 0, 0, 0, 0, 0.6]]))
        assert not prop.valid_input(tensor([[0.25, -0.25, 0.75, 0, 0, 0, 0, 0.6]]))
        assert not prop.valid_input(tensor([[0.33, -0.99, 0, -1, 0, -0.48, 0.25, 1]]))

    def test_robustness_property_factory(self):
        factory = RobustnessPropertyFactory()
        factory.name_prefix('fancy robustness property 💅')
        factory.eps(5e-2)
        factory.desired_extremum('strict_maximum')

        np.random.seed(220321)
        input_samples = np.random.randint(-100, 100, (20, 5))
        labels = [randint(0, 17) for _ in range(20)]

        properties = list(factory.get_properties(input_samples, labels))

        assert len(properties) == 20, 'too little properties generated'
        for index, prop in enumerate(properties):
            assert len(prop.lower_bounds) == 5, 'property has wrong amount of lower bounds'
            assert len(prop.upper_bounds) == 5, 'property has wrong amount of upper bounds'
            assert isinstance(prop.output_constraint, ExtremumConstraint), 'output constraint is not ExtremumConstraint'
            assert prop.output_constraint.output_index == labels[index], 'output constraint has wrong label'

        print(labels)
        dump_specification(properties, stream=sys.stdout)


if __name__ == '__main__':
    unittest.main()
