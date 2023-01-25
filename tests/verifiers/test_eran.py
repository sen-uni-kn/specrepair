import time
import unittest
import random

import numpy as np
import torch
import dill

from deep_opt import NeuralNetwork, Property, BoxConstraint, ExtremumConstraint
from constraint_utils import get_constraints_from_file
from nn_repair.verifiers import ERAN
from nn_repair.verifiers.eran import property_constraint_to_eran, conv1d_to_conv2d

from properties import property_1, property_2, property_8, property_9

# enable all logging to have details when running the tests
import logging
logging.basicConfig(level=logging.DEBUG)


class TestERAN(unittest.TestCase):
    """
    This unit test class contains expensive unit test,
    to check if ERAN works correctly.
    In particular the bridging between the ERAN code and the nn-repair
    code is to be checked.<br>
    A few potentially relevant properties or networks are not included,
    because they take too long to execute.
    """

    def test_acasxu_property1_conversion_to_eran_format(self):
        """
        Cheap test that compares the results of transferring deep_opt property 8 to an eran constraint
        and the result of loading the same property from eran
        """
        prop = property_1()
        converted = property_constraint_to_eran(
            prop.output_constraint, 5,
            np.array([7.5189] * 5), np.array([373.9499] * 5)
        )
        from_eran = get_constraints_from_file('../../eran/data/acasxu/specs/acasxu_prop_1_constraints.txt')

        assert converted[0][0][0] == from_eran[0][0][0]
        assert converted[0][0][1] == from_eran[0][0][1]
        self.assertAlmostEqual(converted[0][0][2], from_eran[0][0][2], 5)

    def test_conv1d_to_conv2d(self):
        """
        Test whether conversion from Conv1d to Conv2d doesn't change the network.
        """
        torch.manual_seed(27692376942472)
        network = torch.load("../../resources/cmapss/cnn_window_20_1.pyt")
        network.eval()
        test_inputs = torch.rand((100,) + network.inputs_shape)
        test_inputs = network.mins + test_inputs * (network.maxes - network.mins)
        test_inputs = torch.vstack(
            [test_inputs, network.mins.unsqueeze(0), network.maxes.unsqueeze(0)]
        )
        orig_outputs = network(test_inputs)

        converted = conv1d_to_conv2d(network)
        converted_outputs = converted(test_inputs.unsqueeze(-1))

        assert torch.allclose(orig_outputs, converted_outputs)

    def test_acasxu_networks_property1(self):
        """
        An expensive test case that runs verification of
        the ACASXu property 1 for some ACASXu networks.
        Compared with the other tests: rather cheap test
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
        verifier = ERAN()

        for i in range(len(networks)):
            print(f"Verifying Network: {indices[i][0]},{indices[i][1]}")
            counterexamples, status = verifier.find_counterexample(networks[i], prop)
            # since property 1 holds for all ACASXu networks,
            # counterexamples should be empty (None would report an error)
            assert len(counterexamples) == 0, f"Found counterexamples for satisfied property: {counterexamples}"
            # wait to allow multiprocessing to shut down properly
            time.sleep(1)

    def test_acasxu_networks_inverse_property1(self):
        """
        An expensive test case that runs verification of
        the inverse of ACAS Xu property 1 for some ACASXu networks.
        Compared to the other tests: rather cheap test
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
        verifier = ERAN()

        for i in range(len(networks)):
            print(f"Verifying Network: {indices[i][0]},{indices[i][1]}")
            counterexamples, status = verifier.find_counterexample(networks[i], prop)
            # since property 1 holds for all ACASXu networks, the inverse needs to be violated
            assert counterexamples is not None, f"An error occurred: {status}"
            print(f"Found {len(counterexamples)} counterexamples. Status: {status}")
            assert len(counterexamples) > 0, "Did not find counterexamples for violated property"
            # wait to allow multiprocessing to shut down properly
            time.sleep(1)

    def test_acasxu_networks_property2(self):
        """
        An expensive test case that runs verification of
        the ACASXu property 2 for some ACASXu networks (including one where this property holds).
        """

        indices = [(2, 1), (5, 3), (2, 9), (4, 7), (3, 3)]
        networks = []
        for i1, i2 in indices:
            net_file = f"../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/" \
                       f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
            networks.append(NeuralNetwork.load_from_nnet(net_file))

        prop = property_2()
        verifier = ERAN()

        for i in range(len(networks)):
            print(f"Verifying Network: {indices[i][0]},{indices[i][1]}")
            counterexamples, status = verifier.find_counterexample(networks[i], prop)
            print(f"Exit status: {status}")
            # property 2 does not hold for all except one network: 3,3
            if indices[i] == (3, 3):
                assert len(counterexamples) == 0, f"Found counterexamples for satisfied property: {counterexamples}"
            else:
                print(f"Found {len(counterexamples)} counterexamples.")
                assert len(counterexamples) > 0, "Did not find counterexamples for violated property"
            # wait to allow multiprocessing to shut down properly
            time.sleep(1)

    def test_acasxu_property8_conversion_to_eran_format(self):
        """
        Cheap test that compares the results of transferring deep_opt property 8 to an eran constraint
        and the result of loading the same property from eran
        """
        prop = property_8()
        converted = property_constraint_to_eran(
            prop.output_constraint, 5,
            np.array([7.5189] * 5), np.array([373.9499] * 5)
        )
        from_eran = get_constraints_from_file('../../eran/data/acasxu/specs/acasxu_prop_8_constraints.txt')

        assert converted == from_eran

    def test_acasxu_networks_property8(self):
        """
        An expensive test case that runs verification of
        the ACASXu property 8 on network 2,9.
        Compared with the other tests: very expensive test (do not run on pc)
        """
        network = NeuralNetwork.load_from_nnet("../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/"
                                               "ACASXU_run2a_2_9_batch_2000.nnet")
        prop = property_8()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        # property 8 does not hold for network 2,9
        print(f"Found {len(counterexamples)} counterexamples.")
        assert len(counterexamples) > 0, "Did not find counterexamples for violated property"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_acasxu_networks_property9(self):
        """
        An expensive test case that runs verification of
        the ACASXu property 9 on network 3,3.
        """
        network = NeuralNetwork.load_from_nnet("../../deep-opt/resources/acas_xu/nnet/polar_r_t_p_vo_vi/"
                                               "ACASXU_run2a_3_3_batch_2000.nnet")
        prop = property_9()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        # property 9 holds for network 3,3
        assert len(counterexamples) == 0, f"Found counterexamples for satisfied property: {counterexamples}"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_acasxu_modified_network_2_9(self):
        """
        Tests verification of a single halfway repaired network
        for which ERAN did report infeasible model.
        Status 29.4.2021: Test now works fine.
        """
        network = NeuralNetwork.load_legacy_pytorch_model("../resources/acasxu_2_9_modified.pyt")

        prop = property_2()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert status != 'ERROR', "verification failed with error"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_acasxu_modified_network_3_9(self):
        """
        Tests verification of a single halfway repaired network
        for which ERAN reported infeasible model.
        Status 1.5.2021: Test now works fine.
        """
        network = NeuralNetwork.load_legacy_pytorch_model("../resources/acasxu_3_9_modified.pyt")

        prop = property_2()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert status != 'ERROR', "verification failed with error"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_acasxu_modified_network_3_5(self):
        """
        Tests verification of a single halfway repaired network
        for which ERAN reports infeasible model.
        Status 1.5.2021: Test now works fine.
        """
        network = NeuralNetwork.load_legacy_pytorch_model("../resources/acasxu_3_5_modified.pyt")

        prop = property_2()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert status != 'ERROR', "verification failed with error"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_acasxu_modified_network_3_8(self):
        """
        Tests verification of a single halfway repaired network
        for which ERAN reports infeasible model.
        Status 1.5.2021: Test now works fine.
        """
        network = NeuralNetwork.load_legacy_pytorch_model("../resources/acasxu_3_8_modified.pyt")

        prop = property_2()
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert status != 'ERROR', "verification failed with error"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    # def test_hcas_modified_2_1(self):
    #     """
    #     Tests verification of a single halfway repaired network
    #     for which ERAN reported violation without returning a counterexample
    #     Status 17.5.2021: Test now works fine, with a larger max_depth
    #                       Suggestion by Mark Müller (https://github.com/eth-sri/eran/issues/73)
    #                       did not work.
    #     """
    #     network = NeuralNetwork.load_legacy_pytorch_model("../resources/hcas_2_1_modified.pyt")

    #     prop = modified_property2
    #     verifier = ERAN(max_depth=15)

    #     counterexamples, status = verifier.find_counterexample(network, prop)
    #     print(f'status={status}, counterexamples={counterexamples}')
    #     assert counterexamples is not None, "verification failed with error"
    #     # wait to allow multiprocessing to shut down properly
    #     time.sleep(1)

    # counterexample generation in analyze_box was disabled again
    # def test_counterexample_generation_analyzer(self):
    #     """
    #     Tests the counterexample generation by analyze_box (ERAN).
    #     """
    #     # two neuron test is a network of two input neurons and one output neuron without activation function
    #     # that simply adds together the two inputs, each with weight 0.5.
    #     model, _ = read_onnx_net('../resources/two_inputs_test_net.onnx')
    #     specLB = [0.0, 0.0]
    #     specUB = [1.0, 1.0]
    #     constraints = [[(0, -1, 0.5)]]  # y0 <= 0.5
    #
    #     eran = ERAN(model, is_onnx=True)
    #     hold, nn, nlb, nub, _, x = eran.analyze_box(specLB, specUB, "deeppoly", 1, 1, True, constraints)
    #     print(x)
    #     assert not hold
    #     assert x is not None, "analyze_box did not return a counterexample"

    # It seems like this network does not actually violate the specification?
    # def test_conv_network_robustness(self):
    #     # this network is trained to classify an all zero input as zero
    #     network = torch.load('../resources/test_conv_network_8x8_inputs.pyt')
    #     prop = Property(
    #         lower_bounds=dict([(i, -0.05) for i in range(64)]),
    #         upper_bounds=dict([(i,  0.05) for i in range(64)]),
    #         output_constraint=ExtremumConstraint(0, '==', 'strict_max'),
    #         property_name='robust at zero'
    #     )
    #     verifier = ERAN()
    #
    #     counterexamples, status = verifier.find_counterexample(network, prop)
    #     print(status)
    #     assert counterexamples is not None and len(counterexamples) > 0, \
    #         "No counterexamples found"

    def test_two_neuron_test_net(self):
        """
        Tests that a counterexample is generated by ERAN for the two neuron test net.
        """
        network = torch.load('../resources/two_inputs_test_net.pyt')

        prop = Property({0: 0, 1: 0}, {0: 1, 1: 1}, BoxConstraint(0, '<=', 0.5))
        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert counterexamples is not None and len(counterexamples) > 0, \
            "No counterexamples found for violated property of trivial network"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)

    def test_eran_mnist_relu_5_100(self):
        network = torch.load('../../resources/mnist/eran_mnist_relu_5_100.pyt')
        # take the first property from the robustness specification with radius 03
        with open('../../resources/mnist/eran_mnist_relu_5_100_not_robust_03.dill', 'rb') as spec_file:
            prop = dill.load(spec_file)[0]

        verifier = ERAN()

        counterexamples, status = verifier.find_counterexample(network, prop)
        print(status)
        assert counterexamples is not None and len(counterexamples) > 0, \
            "No counterexamples found for violated property of MNIST network"
        # wait to allow multiprocessing to shut down properly
        time.sleep(1)
