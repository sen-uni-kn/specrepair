from deep_opt import NeuralNetwork
from deep_opt.models import diff_approx as nn_layers


def remove_terminal_relu(network):
    if isinstance(network[-1], nn_layers.ReLU):
        return NeuralNetwork(
            mins=network.mins, maxes=network.maxes,
            means_inputs=network.means_inputs, ranges_inputs=network.ranges_inputs,
            means_outputs=network.means_outputs, ranges_outputs=network.ranges_outputs,
            modules=list(network.modules())[1:-1],  # the first element is the network itself
            inputs_shape=network.inputs_shape,
            outputs_shape=network.outputs_shape
        )
    else:
        return network
