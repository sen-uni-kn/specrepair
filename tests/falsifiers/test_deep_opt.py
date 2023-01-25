# initialising ray here fixes annoying warnings when running the test cases
import ray
ray.init()

import unittest
import random

import torch

from deep_opt import NeuralNetwork, Property, BoxConstraint, ExtremumConstraint, CounterexampleAmount as CxAm
from nn_repair.falsifiers import DeepOpt

from properties import property_1, property_2

# enable all logging to have details when running the tests
import logging
logging.basicConfig(level=logging.DEBUG)


class TestDeepOptFalsifier(unittest.TestCase):
    """
    This unit test class contains expensive unit test,
    to check if DeepOpt works correctly.

    A few potentially relevant properties or networks are not included,
    because they take too long to execute.
    """

    def test_acasxu_networks_property1(self):
        """
        A somewhat expensive test case that runs verification of
        the ACASXu property 1 for some ACASXu networks.
        """

        random.seed(1326)
        indices = tuple(zip(
            random.choices(range(1, 6), k=7),
            random.choices(range(1, 10), k=7)
        ))

        networks = []
        for i1, i2 in indices:
            net_file = f"../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/" \
                       f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
            networks.append(NeuralNetwork.load_from_nnet(net_file))

        prop = property_1()
        falsifier = DeepOpt(how_many_counterexamples=CxAm.SINGLE, it_simple_upsampling=1,
                            n=10,
                            use_gradient_information=True, cutoff=None)

        for i in range(len(networks)):
            print(f"Falsifying Network: {indices[i][0]},{indices[i][1]}")
            counterexamples, status = falsifier.find_counterexample(networks[i], prop)
            # since property 1 holds for all ACASXu networks,
            # counterexamples should be empty (None would report an error)
            assert len(counterexamples) == 0, f"Found counterexamples for satisfied property: {counterexamples}"
            # wait to allow ray to shutdown properly

    def test_acasxu_networks_inverse_property1(self):
        """
        A somewhat expensive test case that runs verification of
        the inverse of ACASXu property 1 for some ACASXu networks.
        """

        random.seed(2203)
        indices = tuple(zip(
            random.choices(range(1, 6), k=7),
            random.choices(range(1, 10), k=7)
        ))

        networks = []
        for i1, i2 in indices:
            net_file = f"../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/" \
                       f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
            networks.append(NeuralNetwork.load_from_nnet(net_file))

        prop1 = property_1()
        prop = Property(prop1.lower_bounds, prop1.upper_bounds, BoxConstraint(0, '>', 1500),
                        property_name='not ACASXu φ1')
        falsifier = DeepOpt(how_many_counterexamples=CxAm.SINGLE, use_gradient_information=True)

        print(f"Falsifying Networks: {indices}")
        counterexamples_for_each_network = [falsifier.find_counterexample(net, prop) for net in networks]
        print(f"Results: {counterexamples_for_each_network}")
        assert all(counterexamples is not None for counterexamples in counterexamples_for_each_network), \
            "Falsification failed for some network"
        # the test below rather tries to assure the quality of DeepOpt for finding counterexamples
        # the test is not a hard requirement, but it would be desirable that DeepOpt finds counterexamples
        # for at least one network
        assert any(len(counterexamples) > 0 for counterexamples in counterexamples_for_each_network), \
            "No counterexamples found for all network"

    def test_acasxu_networks_property2(self):
        """
        A somewhat expensive test case that runs verification of
        the ACASXu property 2 for some ACASXu networks (including one where this property holds).
        """

        indices = [(2, 1), (5, 3), (2, 9), (4, 7), (3, 3)]
        networks = []
        for i1, i2 in indices:
            net_file = f"../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/" \
                       f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
            networks.append(NeuralNetwork.load_from_nnet(net_file))

        prop = property_2()
        falsifier = DeepOpt(how_many_counterexamples=CxAm.SINGLE, it_simple_upsampling=1,
                            it_peak_sampling=1, it_split_sampling=1,
                            use_gradient_information=True, cutoff=None)

        counterexamples_for_each_network = []
        for i in range(len(networks)):
            print(f"Falsifying Network: {indices[i][0]},{indices[i][1]}")
            counterexamples, status = falsifier.find_counterexample(networks[i], prop)
            print(f"Exit status: {status}")
            assert counterexamples is not None, "Falsification failed"
            # property 2 does not hold for all except one network: 3,3
            if indices[i] == (3, 3):
                assert len(counterexamples) == 0, f"Found counterexamples for satisfied property: {counterexamples}"
            else:
                counterexamples_for_each_network.append(counterexamples)
        # the test below rather tries to assure the quality of DeepOpt for finding counterexamples
        # the test is not a hard requirement, but it would be desirable that DeepOpt finds counterexamples
        # for at least one network
        assert any(len(counterexamples) > 0 for counterexamples in counterexamples_for_each_network), \
            "No counterexamples found for all networks"

    # It seems like this network does not actually violate the specification?
    # def test_conv_network_robustness(self):
    #     # this network is trained to classify an all-zero input as zero
    #     network = torch.load('../resources/test_conv_network_8x8_inputs.pyt')
    #     prop = Property(
    #         lower_bounds=dict([(i, -0.05) for i in range(64)]),
    #         upper_bounds=dict([(i,  0.05) for i in range(64)]),
    #         output_constraint=ExtremumConstraint(0, '==', 'strict_max'),
    #         property_name='robust at zero'
    #     )
    #     falsifier = DeepOpt(how_many_counterexamples=CxAm.SINGLE, it_simple_upsampling=1,
    #                         it_peak_sampling=1, it_split_sampling=1,
    #                         use_gradient_information=True)
    #
    #     counterexamples, status = falsifier.find_counterexample(network, prop)
    #     print(status)
    #     assert counterexamples is not None and len(counterexamples) > 0, \
    #         "No counterexamples found for violated property of trivial network"

    def test_two_neuron_test_net(self):
        """
        Tests that a counterexample is generated by DeepOpt for the two neuron test net.
        """
        network = torch.load('../resources/two_inputs_test_net.pyt')

        prop = Property({0: 0, 1: 0}, {0: 1, 1: 1}, BoxConstraint(0, '<=', 0.5))
        falsifier = DeepOpt(how_many_counterexamples=CxAm.SINGLE, it_simple_upsampling=1,
                            it_peak_sampling=1, it_split_sampling=1,
                            use_gradient_information=True)

        counterexamples, status = falsifier.find_counterexample(network, prop)
        print(status)
        assert counterexamples is not None and len(counterexamples) > 0, \
            "No counterexamples found for violated property of trivial network"
