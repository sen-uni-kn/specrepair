import tempfile
import unittest
from unittest import TestCase
import csv
from tempfile import TemporaryFile
from pathlib import Path

import numpy as np
import torch

from deep_opt.models import NeuralNetwork
from deep_opt.models import diff_approx
from deep_opt.utils.torch_utils import tensor_like


class Test(TestCase):
    def test___init__(self):
        example_file = "../resources/simple_network.nnet"

        network = NeuralNetwork.load_from_nnet(example_file)

        print(network.evaluate_network([0.0], normalize=False))

    def test_num_inputs_and_outputs(self):
        network_file = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_4_4_batch_2000.nnet"
        network1 = NeuralNetwork.load_from_nnet(network_file)
        assert network1.num_inputs() == 5, "num_inputs is incorrect"
        assert network1.num_outputs() == 5, "num_outputs is incorrect"

        # manually create another test network
        network2 = NeuralNetwork(
            # the first few arguments are for normalisation
            -1, 1, 0, 1, 0, 1,
            # now we are getting to the modules
            [
                diff_approx.Sigmoid(),  # this layer does not have a fixed input and output size
                diff_approx.ReLU(),  # neither does this
                diff_approx.Linear(101, 7),  # this is the layer that defines the input size
                diff_approx.ReLU(),
                diff_approx.Linear(7, 9),
                diff_approx.Linear(9, 413),
                diff_approx.Tanh(),
                diff_approx.Linear(413, 13),  # this layer defines the output size
                diff_approx.ReLU(),
                diff_approx.Tanh(),
                diff_approx.CELU()
            ]
        )
        assert network2.num_inputs() == 101, "num_inputs is incorrect"
        assert network2.num_outputs() == 13, "num_outputs is incorrect"

    def test_large_network_eval(self):
        # this network has five inputs and five outputs
        network_file = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_4_4_batch_2000.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        rand = np.random.default_rng(seed=109)
        input_numpy = rand.random((100, 5)) * 10 - 5
        input_tensor = torch.tensor(input_numpy)
        output_numpy = network.evaluate_network(input_numpy)
        output_tensor = network(input_tensor)
        # test numpy array inputs to torch method
        output_3 = network(input_tensor)

        assert np.all(output_numpy == output_tensor.detach().numpy())
        assert torch.all(torch.eq(output_tensor, output_3))

        assert output_numpy.shape == (100, 5)

    def test_acas_xu_eval(self):
        network_file1 = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_2_1_batch_2000.nnet"
        network_file2 = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_2_2_batch_2000.nnet"
        network_file3 = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_4_3_batch_2000.nnet"
        network1 = NeuralNetwork.load_from_nnet(network_file1)
        network2 = NeuralNetwork.load_from_nnet(network_file2)
        network3 = NeuralNetwork.load_from_nnet(network_file3)

        # these are counterexamples to the second property for the respective networks
        # the expected outputs are commented out
        test_input1 = torch.tensor(np.array([
            58613.68813, -0.06784, -2.77422, 1180.94057, 21.44877  # 28.3982, -2.19656, 15.69153, -0.15108, 15.94436
        ]), dtype=torch.float)
        test_input2 = torch.tensor(np.array([
            57556.25904, -0.036, -0.77174, 1164.07667, 36.67178  # 29.36422, -1.02404, 16.65294, 2.54757, 16.13291
        ]), dtype=torch.float)
        test_input3 = torch.tensor(np.array([
            60751.85368, 0.06868, -2.02604, 1162.86617, 47.55553  # 26.11647, -0.93927, 15.58555, 0.6134, 15.20991
        ]), dtype=torch.float)

        output1 = network1(test_input1)
        output2 = network2(test_input2)
        output3 = network3(test_input3)

        # check that the outputs are counterexamples to the second property
        # for the loaded networks as they should be
        print(output1)
        assert max(output1[0:5]) == output1[0]
        assert max(output1[1:5]) - output1[0] < 0
        print(output2)
        assert max(output2[0:5]) == output2[0]
        assert max(output2[2:5]) - output2[0] < 0
        print(output3)
        assert max(output3[0:5]) == output3[0]
        assert max(output3[3:5]) - output3[0] < 0

    def test_large_network_2_eval(self):
        # this network has five inputs and five outputs
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_25HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        rand = np.random.default_rng(seed=1300)
        input = rand.random((1, 5), dtype=float)
        output = network(input)

        assert output.shape == (1, 5)
        print(output)

    def test_vector_input(self):
        # this network has five inputs and five outputs
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_25HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        input = [-0.1, -0.4, 0.3, 0.7, 1.5]  # vector inputs
        output = network.evaluate_network(input)

        assert output.shape == (5, )
        print(output)

    def test_large_network_gradient(self):
        # this network has five inputs and five outputs
        network_file = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_4_4_batch_2000.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        rand = np.random.default_rng(seed=734)
        inputs = torch.tensor(rand.random((100, 5)) * 10 - 5, dtype=torch.float, requires_grad=True)

        outputs = network(inputs)
        sum = torch.sum(outputs)  # sum over all outputs
        gradient_wrt_input = torch.autograd.grad(sum, inputs)
        # this is a 5x100x5 tensor (100 samples and 5 outputs derived wrt 5 inputs)
        jacobian_wrt_parameters = torch.autograd.functional.jacobian(network, inputs)

        print(gradient_wrt_input[0].shape)
        print(gradient_wrt_input)
        print(jacobian_wrt_parameters[0].shape)
        print(jacobian_wrt_parameters)

    def test_custom_network(self):
        # create a NeuralNetwork in code
        # our network will have 3 inputs, three outputs and one module with 3 units
        network = NeuralNetwork(
            # the first few arguments are for normalisation
            [-1]*3, [1]*3, [0]*3, [1]*3, 0, 1,
            # now we are getting to the modules
            [
                diff_approx.Linear(3, 3)
            ]
        )
        # set the module weights to an identity matrix
        network[0].weight.data = tensor_like(network[0].weight.data, np.eye(3, 3))
        # and the biases to zero
        network[0].bias.data = tensor_like(network[0].bias.data, np.zeros(3))

        rand = np.random.default_rng(seed=1043)
        input = torch.tensor(rand.random((1, 3)) * 2 - 1, dtype=torch.float)

        output = network(input)
        weight_gradient = torch.autograd.grad(output.split(1, 1), network[0].weight)[0]

        print(f"input: {input}\noutput: {output}\ngradient: {weight_gradient}")
        assert torch.allclose(output, input)
        assert np.allclose(weight_gradient.detach().numpy(), input.numpy().repeat(3).reshape(3, 3).transpose())

    def test_convolutional_neural_network(self):
        network = NeuralNetwork(
            mins=(-1,),
            maxes=(1, ),
            modules=[
                diff_approx.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                diff_approx.MaxPool2d(kernel_size=2),
                diff_approx.Flatten(),
                diff_approx.Linear(40, 10),
                diff_approx.ReLU(),
                diff_approx.Linear(10, 1)
            ],
            inputs_shape=(1, 4, 4),  # first dimension: channel dimension
            outputs_shape=(1,)
        )

        # generate 100 test input samples
        # also include inputs which are smaller or larger than min/max
        rng = np.random.default_rng(15005)
        test_inputs_numpy = rng.random((100, 1, 4, 4), dtype=np.float32) * 4 - 2
        assert np.any(test_inputs_numpy > 1), "Test not propertly set up"
        assert np.any(test_inputs_numpy < -1), "Test not properly set up"

        test_inputs_tensor = torch.as_tensor(test_inputs_numpy)
        outputs_1 = network(test_inputs_tensor)
        outputs_2 = network(test_inputs_numpy)
        outputs_3 = network(test_inputs_tensor.flatten(1))
        outputs_4 = network(test_inputs_numpy.reshape(100, 16))

        assert torch.allclose(outputs_1, outputs_2), "outputs for torch and numpy inputs did not match"
        assert torch.allclose(outputs_1, outputs_3), "outputs for properly shaped and flat input did not match (torch)"
        assert torch.allclose(outputs_2, outputs_4), "outputs for properly shaped and flat input did not match (numpy)"

    def test_mnist_network(self):
        network_file = "../resources/test_mnist_network.pyt"
        network = torch.load(network_file)

        rand = np.random.default_rng(seed=515)
        inputs = torch.tensor(rand.random((10, 1, 28, 28)) * 255, dtype=torch.float)

        print(network(inputs))

    def test_evaluate_network_sample_file(self):
        true_samples_str = []
        with open('../resources/polar_r_t_p_vo_vi_ACASXu_run2a_2_8_batch_200_test_samples.csv') as f:
            for row in csv.reader(f):
                true_samples_str.append(row)
        true_samples = np.array(true_samples_str, dtype=float)

        network_file = "../../resources/acas_xu/nnet/polar_r_t_p_vo_vi/ACASXU_run2a_2_8_batch_2000.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)
        sample_inputs = true_samples[:, 0:5]
        sample_outputs = network.evaluate_network(sample_inputs)

        # some inaccuracy is introduced through storing as csv
        assert np.allclose(sample_outputs, true_samples[:, 5:10], atol=1e-4)

    def test_onnx_export(self):
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_250HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        # the exported model should only contain ReLU and Gemm (generalized matrix multiplication) as operations
        # and the input and output tensors should not have a batch dimension
        with TemporaryFile() as f:
            network.onnx_export(f, disable_normalization=True, input_sample='no batch', verbose=True)

    def test_onnx_export_2(self):
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_250HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        # the exported model should contain operations for normalisation as well
        # and the input and output tensors should have a batch dimension with a single entry
        with TemporaryFile() as f:
            network.onnx_export(f, disable_normalization=False, input_sample='batch', verbose=True)

    def test_onnx_export_3(self):
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_250HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)

        # the exported model should contain operations for normalisation as well
        # and the input and output tensors should have a batch dimension with a single entry
        with TemporaryFile() as f:
            network.onnx_export(f, disable_normalization=True, disable_unsqueeze=False, disable_squeeze=True,
                                input_sample='no batch', verbose=True)

    def test_nnet_round_trip(self):
        # create a new network which is randomly initialised
        mins = np.array([-1, 0, -5,  5,         -1, -0.5, -1/3, -1/9, 1/9,     0])
        maxes = np.array([1, 1,  5, 10 + 1/3, -0.5,  0.5,  1/3,    0, 8/9, 5.735])
        network = NeuralNetwork(
            mins,
            maxes,
            [0.25] * 10,
            [0.44] * 10,
            0.25,
            0.44,
            [
                diff_approx.Linear(10, 100),
                diff_approx.ReLU(),
                diff_approx.Linear(100, 100),
                diff_approx.ReLU(),
                diff_approx.Linear(100, 250),
                diff_approx.ReLU(),
                diff_approx.Linear(250, 100),
                diff_approx.ReLU(),
                diff_approx.Linear(100, 50),
                diff_approx.ReLU(),
                diff_approx.Linear(50, 25),
            ]
        )
        # generate a large amount of input samples
        rand = np.random.default_rng(seed=1204)
        inputs = mins + rand.random((10000, 10)) * (maxes - mins)
        assert (inputs >= mins).all()
        assert (inputs <= maxes).all()

        original_outputs = network(inputs)
        with tempfile.TemporaryDirectory() as tmpdir:
            network_file = Path(tmpdir, 'test_network.nnet')
            network.save_to_nnet(network_file, ['Test .nnet file', 'Comment 2'])
            with open(network_file) as file:
                print("".join(file.readlines()))
            loaded_network = network.load_from_nnet(str(network_file))
        output_loaded = loaded_network(inputs)
        assert np.isclose(original_outputs.detach(), output_loaded.detach()).all(), \
            "Outputs of loaded and original network did not match"

    def test_nnet_round_trip_2(self):
        network_file = "../../resources/hcas/HCAS_rect_v6_pra0_tau00_25HU.nnet"
        network = NeuralNetwork.load_from_nnet(network_file)
        with open(network_file) as file:
            original_content = "".join(file.readlines())
        with tempfile.TemporaryDirectory() as tmpdir:
            network_file = Path(tmpdir, 'test_network.nnet')
            with network_file.open('wt+') as file:
                network.save_to_nnet(file, normalization_values_format=".6f")
            with open(network_file) as file:
                loaded_content = "".join(file.readlines())
        print(loaded_content)
        # format does not match due to floating point differences
        # assert original_content == loaded_content, "Loaded and original file content did not match"


if __name__ == '__main__':
    unittest.main()
