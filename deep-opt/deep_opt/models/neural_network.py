from typing import List, Optional, Union, Any, Tuple, IO, Sequence

from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch import nn
from math import prod

import onnx
import onnx2pytorch

import deep_opt.models.differentiably_approximatable_nn_modules as diff_approx
import deep_opt.utils.legacy_loading
from deep_opt.utils.torch_utils import tensor_like


class NeuralNetwork(torch.nn.Sequential):
    """
    Class that represents a neural network for use in DeepOpt.
    This class allows approximating transformation functions such as ReLU
    that are not differentiable everywhere with other transformation functions
    that are differentiable everywhere.

    It also allows reading and writing neural networks to and from .nnet files (Kyle Julian 2016)
    that such that they can be used in torch.

    Approximation NeuralNetworks created by this class do not share weights with the
    original network (unless the approximation is the same as the original network if
    that is already differentiable everywhere).

    Due to missing onnx support, NeuralNetworks use the single precision or float32
    dtype for all operations. This is the standard type in torch, but not in numpy.
    NeuralNetwork will do it's best to convert numpy array to the float type where it sees them.
    """

    def __init__(self,
                 # can handle all types that torch.tensor can handle
                 mins: Union[torch.Tensor, np.ndarray, Any] = -float('inf'),
                 maxes: Union[torch.Tensor, np.ndarray, Any] = float('inf'),
                 means_inputs: Union[torch.Tensor, np.ndarray, Any] = 0,
                 ranges_inputs: Union[torch.Tensor, np.ndarray, Any] = 1,
                 means_outputs: Union[torch.Tensor, np.ndarray, Any] = 0,
                 ranges_outputs: Union[torch.Tensor, np.ndarray, Any] = 1,
                 modules: Optional[Union[Tuple, List]] = None,
                 inputs_shape: Optional[Tuple[int, ...]] = None,
                 outputs_shape: Optional[Tuple[int, ...]] = None):
        """
        Creates a new NeuralNetwork with the given modules and normalisation parameters.

        The default values of the normalisation parameters effectively disable normalisation.

        :param mins: The minimum value of the inputs.
         Smaller inputs are clipped to the respective minimum.
         A vector of the shape of the inputs or broadcastable to the inputs' shape.
        :param maxes: The maximum value of the inputs. Larger inputs are clipped to the respective maximum.
         A vector of the shape of the inputs or broadcastable to the inputs' shape.
        :param means_inputs: The mean value of each input.
         During normalisation this mean value is shifted to 0 for each input dimension.
         A value of 0 disables the normalisation.
         A vector of the shape of the inputs or broadcastable to the inputs' shape.
        :param ranges_inputs: The ranges of each input.
         During normalisation this range is transformed to 1 for each input dimension.
         A value of 1 disables the normalisation.
         A vector of the shape of the inputs or broadcastable to the inputs' shape.
        :param means_outputs: The mean value of each output.
         During normalisation this mean value is shifted to 0 for each output dimension.
         A value of 0 disables the normalisation.
         A vector of the shape of the outputs or broadcastable to the outputs' shape.
        :param ranges_outputs: The ranges of each output.
         During normalisation this range is transformed into one for each output dimension.
         A value of 1 disables the normalisation.
         A vector of the shape of the outputs or broadcastable to the outputs' shape.
        :param modules: The modules/layers of the created NeuralNetwork
        :param inputs_shape: The shape of the inputs. If None is passed, NeuralNetwork tries to infer the
         input shape based on the modules of the network.
         However, this may yield an inputs' shape with wildcards (-1s), which do not allow data generation.
         To allow generating data, supply an input shape here.
         For more details on the shape inference see the inputs_shape property.
        :param outputs_shape: The shape of the outputs. If None is passed, NeuralNetwork tries to infer the
         output shape based on the modules of the network.
         However, this may yield an outputs' shape containing wildcards (-1s).
         For more details on the shape inference, see the outputs_shape property.
        """

        # super will call add_modules which will check if the module is supported
        super().__init__(*modules)
        self.float()  # this is actually default

        self.fixed_inputs_shape = inputs_shape
        self.fixed_outputs_shape = outputs_shape
        # infer inputs and outputs shapes to expand the mins, maxes,
        # means and ranges
        inputs_shape = self.inputs_shape
        outputs_shape = self.outputs_shape

        mins = torch.as_tensor(mins, dtype=torch.float).expand(inputs_shape).clone()
        self.register_buffer("mins", mins)
        maxes = torch.as_tensor(maxes, dtype=torch.float).expand(inputs_shape).clone()
        self.register_buffer("maxes", maxes)
        means_inputs = torch.as_tensor(means_inputs, dtype=torch.float).expand(inputs_shape).clone()
        self.register_buffer("means_inputs", means_inputs)
        ranges_inputs = torch.as_tensor(ranges_inputs, dtype=torch.float).expand(inputs_shape).clone()
        self.register_buffer("ranges_inputs", ranges_inputs)
        means_outputs = torch.as_tensor(means_outputs, dtype=torch.float).expand(outputs_shape).clone()
        self.register_buffer("means_outputs", means_outputs)
        ranges_outputs = torch.as_tensor(ranges_outputs, dtype=torch.float).expand(outputs_shape).clone()
        self.register_buffer("ranges_outputs", ranges_outputs)

    def add_module(self, name: str, module: Optional[torch.nn.Module]):
        # check that the added module is supported
        if module is not None:
            self._check_supported_module_type(module)
        super().add_module(name, module=module)

    def __setattr__(self, key: str, value: Union[torch.Tensor, torch.nn.Module]):
        if isinstance(value, torch.nn.Module):
            self._check_supported_module_type(value)
        super().__setattr__(key, value)

    def _check_supported_module_type(self, module: torch.nn.Module) -> None:
        """
        Makes sure a given module is supported by DeepOpt.
        This requires the module to be either differentiable everywhere (e.g. ReLU is not)
        or to be approximatable by a differentiable function.
        This is made sure by only allowing `DiffApproxModule` instances.
        This function will throw an error if the given module is not supported.
        """
        pass

    def forward(self, inputs, normalize=True, disable_unsqueeze=False, disable_squeeze=False, disable_unflatten=False):
        """
        Evaluate the network with the given inputs.
        :param inputs: The inputs to the network.
        Can be a Tensor, numpy array or any type that can be converted
        with torch.tensor to a Tensor.
        The input tensor may also be flat despite the network needing a multidimensional input if the inputs_shape
        is clearly defined (not None, no -1 wildcards).
        In this case flat inputs will be unflattened to the inputs_shape.
        :param normalize: Whether to apply input normalization and output renormalization
         with this models mins, maxes, means and ranges.
        :param disable_unsqueeze: Dot not apply the unsqueeze operation that is necessary for using forward
        with an input tensor without batch dimension. Option for controlling the onnx export.
        :param disable_squeeze: Dot not apply the squeeze operation that is applied to return vector output
        if the input tensor has not batch dimension (is a vector). Option for controlling the onnx export.
        :param disable_unflatten: Do not reshape flat input vectors to the inputs_shape. Option for controlling
        the onnx export.
        :return: The networks output, as a Tensor.
        """
        tensor_inputs: torch.Tensor = inputs
        if isinstance(inputs, np.ndarray):
            tensor_inputs = torch.as_tensor(inputs, dtype=torch.float)
        elif not isinstance(inputs, torch.Tensor):
            tensor_inputs = torch.tensor(inputs, dtype=torch.float)

        if tensor_inputs.device != self.mins.device:
            tensor_inputs = tensor_inputs.to(self.mins.device)

        vector_inputs = tensor_inputs.dim() == 1
        if vector_inputs and not disable_unsqueeze:
            tensor_inputs = tensor_inputs.unsqueeze(0)

        # if the networks inputs_shape is multidimensional,
        # but the given input is two dimensional (batch dimension + 1)
        # then reshape the input to the right inputs_shape
        if (
            not disable_unflatten
            and tensor_inputs.ndim == 2
            and self.inputs_shape is not None
            and all(d >= 0 for d in self.inputs_shape)
        ):
            tensor_inputs = tensor_inputs.reshape(
                (tensor_inputs.shape[0],) + self.inputs_shape
            )

        # make the input a float tensor if necessary
        # put this in an if branch, because otherwise on ONNX export, a cast operation
        # would be exported that ERAN doesn't support
        if tensor_inputs.dtype != torch.float:
            tensor_inputs = tensor_inputs.float()

        # normalize
        norm_inputs = tensor_inputs
        if normalize:
            # clip to lower bound (mins)
            # a simple way to do this would be the line below, but onnx does not support this operation
            # clipped_inputs = torch.maximum(tensor_inputs, self.mins.type_as(tensor_inputs))
            mins_broadcasted = self.mins.expand_as(tensor_inputs)
            clipped_inputs = torch.max(torch.stack([tensor_inputs, mins_broadcasted], 0), 0).values
            # clip to upper bound (maxes) (again onnx doesn't support minimum
            # clipped_inputs = torch.minimum(clipped_inputs, self.maxes.type_as(tensor_inputs))
            maxes_broadcasted = self.maxes.expand_as(tensor_inputs)
            clipped_inputs = torch.min(torch.stack([clipped_inputs, maxes_broadcasted], 0), 0).values
            # - and / do broadcasting themselves
            norm_inputs = (clipped_inputs - self.means_inputs) / self.ranges_inputs

        outputs = super().forward(norm_inputs)

        # Undo output normalization
        if normalize:
            outputs = outputs * self.ranges_outputs + self.means_outputs

        # if the inputs was a vector, reshape the output to also be a vector
        if vector_inputs and not disable_squeeze:
            outputs = outputs.squeeze(dim=0)  # only squeeze the batch dimension
        return outputs

    def evaluate_network(self, inputs: np.array, normalize: bool = True) -> np.array:
        """
        Evaluate network using given inputs.

        The difference between this method and calling a NeuralNetwork is that
        this method also returns a numpy array as output.
        :param inputs: (numpy array of floats) Network inputs to be evaluated.
        This can either be a single inputs vector or multiple inputs vectors as a matrix.
        In such a matrix each inputs vector needs to fill up one row.
        :param normalize: inputs and output normalization
        :return: (numpy array of floats): Network outputs for the given inputs samples.
        """
        output_tensor = self(inputs, normalize=normalize)
        return output_tensor.detach().numpy()

    def is_convolutional(self) -> bool:
        """
        Determines whether this NeuralNetwork has at least one convolutional layer.
        """
        return any(isinstance(module, nn.Conv1d) or
                   isinstance(module, nn.Conv2d) or
                   isinstance(module, nn.Conv3d)
                   for module in self)

    def has_pooling(self) -> bool:
        """
        Determines whether this NeuralNetwork has at least one pooling layer.
        """
        return any(isinstance(module, nn.MaxPool1d) or
                   isinstance(module, nn.MaxPool2d) or
                   isinstance(module, nn.MaxPool3d) or
                   isinstance(module, nn.AvgPool1d) or
                   isinstance(module, nn.AvgPool2d) or
                   isinstance(module, nn.AvgPool3d)
                   for module in self)

    def is_fully_connected(self) -> bool:
        """
        Determines whether this NeuralNetwork has only linear, activation and flatten layers.
        """
        return all(isinstance(module, nn.Linear) or
                   isinstance(module, nn.Flatten) or
                   isinstance(module, nn.ReLU) or
                   isinstance(module, nn.Sigmoid) or
                   isinstance(module, nn.Tanh) or
                   isinstance(module, nn.LeakyReLU) or
                   isinstance(module, nn.CELU) or
                   isinstance(module, nn.GELU) or
                   isinstance(module, nn.SELU) or
                   isinstance(module, nn.ELU) or
                   isinstance(module, nn.Identity)
                   for module in self)

    @property
    def inputs_shape(self) -> Optional[Tuple[int, ...]]:
        """
        The shape of the inputs this NeuralNetwork accepts (without additional first batch dimension).
        This shape is either supplied at construction or otherwise inferred from the modules of the NeuralNetwork.

        Inference works by looking at the modules of the NeuralNetwork one after another.
        The first module which has a inputs_shape that is not None is then returned as the inputs_shape
        of the whole network.
        Note that this shape may contain wildcards (-1s), despite not all possible values for these wildcard
        dimensions may be supported.
        """
        if self.fixed_inputs_shape is not None:
            return self.fixed_inputs_shape
        else:
            for module in self:
                if module.inputs_shape is not None:
                    return module.inputs_shape
            # This network can be applied to inputs with any shape
            return None

    def num_inputs(self) -> Optional[int]:
        """
        Returns the flattened inputs shape of this network.
        Returns None if the inputs shape is None or contains wildcards.
        """
        inputs_shape = self.inputs_shape
        if inputs_shape is None or any(s < 0 for s in inputs_shape):
            return None
        else:
            return prod(inputs_shape)

    @property
    def outputs_shape(self) -> Optional[Tuple[int, ...]]:
        """
        The shape of the outputs this NeuralNetwork produces (without additional first batch dimension).
        This shape is either supplied at construction or otherwise inferred from the modules of the NeuralNetwork.

        Inference works by looking at the modules of the NeuralNetwork one after another from the rear end.
        The first module which has a outputs_shape that is not None is then returned as the outputs_shape
        of the whole network.
        Note that this shape may contain wildcards (-1s), despite only some or one fixed value being produced
        if the concrete value depends on earlier modules.
        """
        if self.fixed_outputs_shape is not None:
            return self.fixed_outputs_shape
        else:
            # here iterate starting at the end, then take the first fixed output dimension
            for module in reversed(self):
                if module.outputs_shape is not None:
                    return module.outputs_shape
            # Network output shape is dependent on the inputs shape
            return None

    def num_outputs(self) -> Optional[int]:
        """
        Returns the flattened outputs shape of this network.
        Returns None if the outputs shape is None or contains wildcards.
        """
        outputs_shape = self.outputs_shape
        if outputs_shape is None or any(s < 0 for s in outputs_shape):
            return None
        else:
            return prod(outputs_shape)

    def get_bounds(self, epsilons: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """
        Returns the flattened input bounds (mins and maxes)

        :param epsilons: allows growing the bounds by some constant. A value of None disables this option.
        :return: A list of tuples. Each tuple gives as first element the lower bound of one input element
         and as second element the upper bound for the same input element.
        """
        if epsilons is None:
            epsilons = [0.0, 0.0]

        bounds = []
        mins = self.mins.flatten()
        maxes = self.maxes.flatten()
        for i in range(self.num_inputs()):
            bounds.append((mins[i] - epsilons[0], maxes[i] + epsilons[1]))

        return bounds

    @staticmethod
    def load_from_nnet(filename: str) -> 'NeuralNetwork':
        """
        Loads ReLU activated neural network model from a .nnet file.
        :param filename: The path to the .nnet file
        :return: A new NeuralNetwork with the architecture and parameters stored in the given file
        """
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line = f.readline()
                cnt += 1
            # numLayers doesn't include the inputs module!
            num_layers, input_size, output_size, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()

            # inputs module size, layer1size, layer2size...
            layer_sizes = [int(x) for x in line.strip().split(",")[:-1]]

            # the next line contains a flag that is not use; ignore
            f.readline()
            # symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            input_minimums = [float(x) for x in line.strip().split(",") if x != ""]

            while len(input_minimums) < input_size:
                input_minimums.append(min(input_minimums))

            line = f.readline()
            input_maximums = [float(x) for x in line.strip().split(",") if x != ""]

            while len(input_maximums) < input_size:
                input_maximums.append(max(input_maximums))

            line = f.readline()
            means = [float(x) for x in line.strip().split(",")[:-1]]

            # if there are too little means given (we also need one for the output)
            # fill up with 0, which will cause no modifications in the data
            if len(means) < input_size + 1:
                means.append(0.0)

            line = f.readline()
            ranges = [float(x) for x in line.strip().split(",")[:-1]]

            # same as with means
            if len(ranges) < input_size + 1:
                ranges.append(1.0)

            weights = []
            biases = []

            # Matrix of Neural Network
            #
            # The first dimension will be the module number
            # The second dimension will be 0 for weights, 1 for biases
            # The third dimension will be the number of neurons in that module
            # The fourth dimension will be the number of inputs to that module
            #
            # Note that the bias array will have only number per neuron, so
            # its fourth dimension will always be one
            for layer_idx in range(num_layers):
                previous_layer_size = layer_sizes[layer_idx]
                current_layer_size = layer_sizes[layer_idx + 1]

                weights.append([])
                biases.append([])

                weights[layer_idx] = np.zeros((current_layer_size, previous_layer_size))

                for i in range(current_layer_size):
                    line = f.readline()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]

                    for j in range(previous_layer_size):
                        # FIXME: problem with multi-dimensional output in .nnet file
                        weights[layer_idx][i, j] = aux[j]

                # biases
                biases[layer_idx] = np.zeros(current_layer_size)
                for i in range(current_layer_size):
                    line = f.readline()
                    x = float(line.strip().split(",")[0])
                    biases[layer_idx][i] = x

        modules = []
        for i in range(num_layers - 1):  # hidden layers
            linear_layer = diff_approx.Linear(
                in_features=layer_sizes[i],  # layer_sizes[0] contains the number of inputs
                out_features=layer_sizes[i + 1]
            )
            # torch wants shape units x inputs, which is what we already have
            linear_layer.weight.data = tensor_like(linear_layer.weight.data, weights[i])
            linear_layer.bias.data = tensor_like(linear_layer.bias.data, biases[i])
            modules.append(linear_layer)
            modules.append(diff_approx.ReLU())
        # add the output layer
        output_layer = diff_approx.Linear(
            in_features=layer_sizes[-2],
            out_features=layer_sizes[-1]
        )
        output_layer.weight.data = tensor_like(output_layer.weight.data, weights[-1])
        output_layer.bias.data = tensor_like(output_layer.bias.data, biases[-1])
        modules.append(output_layer)

        network = NeuralNetwork(
            input_minimums, input_maximums,
            means_inputs=means[:-1], ranges_inputs=ranges[:-1],
            means_outputs=means[-1], ranges_outputs=ranges[-1],
            modules=modules,
            inputs_shape=(layer_sizes[0], ), outputs_shape=(layer_sizes[-1], )
        )
        return network

    def save_to_nnet(self, file: Union[str, PathLike, IO[str]], comments: Sequence[str] = (),
                     normalization_values_format="g", parameter_format=".5e"):
        """
        Saves a ReLU activated feed-forward neural network model to a .nnet file.
        If the network does not strictly alternate between Linear and ReLU layers or if the last output layer
        is not a linear layer, this method raises a ValueError.
        :param file: Either the path to the place where the .nnet file should be stored (str or PathLike) or
        a IO object, such as created by `open`. If the argument is a string or PathLike,
        the method tries to create a file at this path. The method fails if the file already exists.
        :param comments: Optional comments that will be placed at the top of the file.
        Each element of the sequence corresponds to one line. Newlines will result in invalid output.
        :param normalization_values_format: The format string that will be used to write normalisation values
        (mins, maxes, means and ranges). The default value prints the full float (non lossy) in floating point format
        (not scientific format).
        :param parameter_format: The format string that will be used to write weights and biases to the file.
        The default value is .5e, which prints five digits in scientific format. This is also the value used in the
        NNet repository (https://github.com/sisl/NNet). This however will result in a lossy representation.
        Use ".e" for a non-lossy saving.
        """
        # this method is largely based on a file from the NNet repository by Kyle Julian
        # MIT License; Copyright 2018 Stanford Intelligent Systems Laboratory
        # https://github.com/sisl/NNet/blob/4411dd47621489f44062ca96898b8cebd722c7c8/utils/writeNNet.py

        # check that self can be exported to .nnet
        # first check that all output means and ranges are the same
        outputs_first_index = tuple(0 for _ in self.outputs_shape)
        first_output_mean = self.means_outputs[outputs_first_index]
        first_output_range = self.ranges_outputs[outputs_first_index]
        if torch.any(self.means_outputs != first_output_mean):
            raise ValueError(f"The .nnet format only supports a single mean value for all outputs, "
                             f"but found multiple values: {self.means_outputs}")
        if torch.any(self.ranges_outputs != first_output_range):
            raise ValueError(f"The .nnet format only supports a single range value for all outputs, "
                             f"but found multiple values: {self.means_outputs}")
        # now check that the layers are supported
        if not isinstance(self[-1], diff_approx.Linear):
            raise ValueError("The NeuralNetworks output layer needs to be a Linear layer "
                             "for export to the .nnet format.")
        if not isinstance(self[0], diff_approx.Linear):
            raise ValueError("The NeuralNetworks input layer needs to be a Linear layer "
                             "for export to the .nnet format.")
        current_layer_is_linear = False  # start from a "one before the start" position
        for module in self:
            if not isinstance(module, diff_approx.ReLU) and not isinstance(module, diff_approx.Linear):
                raise ValueError("Only ReLU and Linear layers are permitted by the .nnet format.")
            if current_layer_is_linear and not isinstance(module, diff_approx.ReLU):
                raise ValueError("Network needs to consist of alternating Linear and ReLU layers "
                                 "for saving to the .nnet format.")
            if not current_layer_is_linear and not isinstance(module, diff_approx.Linear):
                raise ValueError("Network needs to consist of alternating Linear and ReLU layers "
                                 "for saving to the .nnet format.")
            current_layer_is_linear = not current_layer_is_linear

        # If file is a string or PathLike, create and open the file first
        path = None
        try:
            if isinstance(file, str) or isinstance(file, PathLike):
                path = Path(file)
                file = path.open('xt')

            # now write the network
            # comments first
            for comment in comments:
                file.write("// " + comment + "\n")
            file.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

            # write sizes of layers, inputs and outputs
            layer_sizes = tuple(module.out_features for module in self if isinstance(module, diff_approx.Linear))
            num_layers = len(layer_sizes)
            file.write("%d,%d,%d,%d,\n" % (num_layers, self.num_inputs(), self.num_outputs(), max(layer_sizes)))
            file.write("%d," % self.num_inputs())
            for layer_size in layer_sizes:
                file.write("%d," % layer_size)
            file.write("\n")
            file.write("0,\n")  # Unused Flag

            # normalisation values (mins, maxes, etc are tensors, hence .item())
            file.write(','.join(("%" + normalization_values_format) % m.item() for m in self.mins) + ',\n')
            file.write(','.join(("%" + normalization_values_format) % m.item() for m in self.maxes) + ',\n')
            file.write(','.join(("%" + normalization_values_format) % m.item() for m in self.means_inputs))
            file.write((',' + ("%" + normalization_values_format) + ',\n') % first_output_mean)
            file.write(','.join(("%" + normalization_values_format) % r.item() for r in self.ranges_inputs))
            file.write((',' + ("%" + normalization_values_format) + ',\n') % first_output_range)

            # weights and biases
            for module in self:
                if isinstance(module, diff_approx.Linear):
                    w = module.weight
                    b = module.bias
                    for i in range(w.shape[0]):
                        for j in range(w.shape[1]):
                            # write only five digits as in the original code
                            file.write(("%" + parameter_format + ",") % w[i][j])
                        file.write("\n")
                    for i in range(b.shape[0]):
                        # also write the whole float value here
                        file.write(("%" + parameter_format + ",\n") % b[i])

        finally:
            if path is not None:
                file.close()

    @staticmethod
    def load_legacy_pytorch_model(file) -> 'NeuralNetwork':
        """
        Load a NeuralNetwork that was saved using ``torch.save``
        before the introduction of convolutional layers and non flat input shapes.

        Such models can not be loaded using ``torch.load`` because the differentiable approximatable
        layers module has been renamed (typo) and they can not really be used, since they
        do not supply an ``inputs_shape`` attribute.
        :param file: The file from which the (old) NeuralNetwork should be loaded.
        Needs to have a type that can be handled by ``torch.load``
        :return: A (new) NeuralNetwork loaded from the given file
        """
        # deep_opt.utils.legacy_loading fixes module names that changed, e.g.
        # - differentiably_approximatable_nn_modules (renamed to fix typo)
        old_network = torch.load(file, pickle_module=deep_opt.utils.legacy_loading)
        return NeuralNetwork(
            mins=old_network.mins, maxes=old_network.maxes,
            # old network format had only one means and ranges array
            # with the last value used for all outputs (like nnet)
            means_inputs=old_network.means[:-1], ranges_inputs=old_network.ranges[:-1],
            means_outputs=old_network.means[-1], ranges_outputs=old_network.ranges[-1],
            # the network can be expanded using *
            modules=old_network,
            # shape inference (resp. num_inputs() and num_outputs())
            # was based only on the modules in the old version
            inputs_shape=None, outputs_shape=None
        )

    @staticmethod
    def load_from_onnx(file, **kwargs) -> 'NeuralNetwork':
        """
        Convert an ONNX network stored in the given file to a NeuralNetwork.

        :param file: The file the ONNX network is stored in.
        :param kwargs: Additional arguments ot the NeuralNetwork constructor, e.g. mins or inputs_shape.
        :return: A NeuralNetwork with the sample modules and parameter values as the stored ONNX network
         and the given additional NeuralNetwork constructor arguments.
        """
        onnx_model = onnx.load(file)
        converted_network = onnx2pytorch.ConvertModel(onnx_model)
        # copy to new model to enable batch processing
        # convert layers to diff_approx layers
        modules = []
        onnx_modules_gen = converted_network.modules()
        next(onnx_modules_gen)  # the first module of converted_network is this network itself
        for module in onnx_modules_gen:
            if isinstance(module, torch.nn.ReLU):
                modules.append(diff_approx.ReLU())
            elif isinstance(module, torch.nn.Tanh):
                modules.append(diff_approx.Tanh())
            elif isinstance(module, torch.nn.Sigmoid):
                modules.append(diff_approx.Sigmoid())
            elif isinstance(module, torch.nn.LogSigmoid):
                modules.append(diff_approx.LogSigmoid())
            elif isinstance(module, torch.nn.CELU):
                modules.append(diff_approx.CELU())
            elif isinstance(module, torch.nn.GELU):
                modules.append(diff_approx.GELU())
            elif isinstance(module, torch.nn.Softsign):
                modules.append(diff_approx.Softsign())
            elif isinstance(module, torch.nn.Tanhshrink):
                modules.append(diff_approx.Tanhshrink())
            elif isinstance(module, torch.nn.Softmin):
                modules.append(diff_approx.Softmin())
            elif isinstance(module, torch.nn.Softmax):
                modules.append(diff_approx.Softmax())
            elif isinstance(module, torch.nn.Softmax2d):
                modules.append(diff_approx.Softmax2d())
            elif isinstance(module, torch.nn.LogSoftmax):
                modules.append(diff_approx.LogSoftmax())
            elif isinstance(module, torch.nn.Identity):
                modules.append(diff_approx.Identity())
            elif isinstance(module, torch.nn.Linear):
                copy = diff_approx.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                copy.weight.data = module.weight.data
                copy.bias.data = module.bias.data if module.bias is not None else None
                modules.append(copy)
            elif isinstance(module, torch.nn.Conv1d):
                copy = diff_approx.Conv1d(module.in_channels, module.out_channels,
                                          module.kernel_size, module.stride,
                                          module.padding, module.dilation,
                                          module.groups, module.bias is not None,
                                          module.padding_mode)
                copy.weight.data = module.weight.data
                copy.bias.data = module.bias.data if module.bias is not None else None
                modules.append(copy)
            elif isinstance(module, torch.nn.Conv2d):
                copy = diff_approx.Conv2d(module.in_channels, module.out_channels,
                                          module.kernel_size, module.stride,
                                          module.padding, module.dilation,
                                          module.groups, module.bias is not None,
                                          module.padding_mode)
                copy.weight.data = module.weight.data
                copy.bias.data = module.bias.data if module.bias is not None else None
                modules.append(copy)
            elif isinstance(module, torch.nn.Conv3d):
                copy = diff_approx.Conv3d(module.in_channels, module.out_channels,
                                          module.kernel_size, module.stride,
                                          module.padding, module.dilation,
                                          module.groups, module.bias is not None,
                                          module.padding_mode)
                copy.weight.data = module.weight.data
                copy.bias.data = module.bias.data if module.bias is not None else None
                modules.append(copy)
            elif isinstance(module, torch.nn.MaxPool1d):
                modules.append(diff_approx.MaxPool1d(
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.return_indices, module.ceil_mode
                ))
            elif isinstance(module, torch.nn.MaxPool2d):
                modules.append(diff_approx.MaxPool2d(
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.return_indices, module.ceil_mode
                ))
            elif isinstance(module, torch.nn.MaxPool3d):
                modules.append(diff_approx.MaxPool3d(
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.return_indices, module.ceil_mode
                ))
            elif isinstance(module, torch.nn.AvgPool1d):
                modules.append(diff_approx.AvgPool1d(
                    module.kernel_size, module.stride, module.padding,
                    module.ceil_mode, module.count_include_pad
                ))
            elif isinstance(module, torch.nn.AvgPool2d):
                modules.append(diff_approx.AvgPool2d(
                    module.kernel_size, module.stride, module.padding,
                    module.ceil_mode, module.count_include_pad,
                    module.divisor_override
                ))
            elif isinstance(module, torch.nn.AvgPool3d):
                modules.append(diff_approx.AvgPool3d(
                    module.kernel_size, module.stride, module.padding,
                    module.ceil_mode, module.count_include_pad
                ))
            elif isinstance(module, torch.nn.Dropout):
                modules.append(diff_approx.Dropout(module.p, module.inplace))
            elif isinstance(module, torch.nn.Dropout2d):
                modules.append(diff_approx.Dropout2d(module.p, module.inplace))
            elif isinstance(module, torch.nn.Dropout3d):
                modules.append(diff_approx.Dropout3d(module.p, module.inplace))
            elif isinstance(module, torch.nn.AlphaDropout):
                modules.append(diff_approx.AlphaDropout(module.p, module.inplace))
            elif isinstance(module, torch.nn.BatchNorm1d):
                copy = diff_approx.BatchNorm1d(
                    module.num_features, module.eps, module.momentum,
                    module.affine, module.track_running_stats
                )
                copy.weight.data = module.weight.data if module.weight is not None else None
                copy.bias.data = module.bias.data if module.bias is not None else None
                copy.running_mean.data = module.running_mean.data if module.running_mean is not None else None
                copy.running_var.data = module.running_var.data if module.running_var is not None else None
                copy.num_batches_tracked.data = module.num_batches_tracked.data if module.num_batches_tracked else None
                modules.append(copy)
            elif isinstance(module, torch.nn.BatchNorm2d):
                copy = diff_approx.BatchNorm2d(
                    module.num_features, module.eps, module.momentum,
                    module.affine, module.track_running_stats
                )
                copy.weight.data = module.weight.data if module.weight is not None else None
                copy.bias.data = module.bias.data if module.bias is not None else None
                copy.running_mean.data = module.running_mean.data if module.running_mean is not None else None
                copy.running_var.data = module.running_var.data if module.running_var is not None else None
                copy.num_batches_tracked.data = module.num_batches_tracked.data if module.num_batches_tracked else None
                modules.append(copy)
            elif isinstance(module, torch.nn.BatchNorm3d):
                copy = diff_approx.BatchNorm3d(
                    module.num_features, module.eps, module.momentum,
                    module.affine, module.track_running_stats
                )
                copy.weight.data = module.weight.data if module.weight is not None else None
                copy.bias.data = module.bias.data if module.bias is not None else None
                copy.running_mean.data = module.running_mean.data if module.running_mean is not None else None
                copy.running_var.data = module.running_var.data if module.running_var is not None else None
                copy.num_batches_tracked.data = module.num_batches_tracked.data if module.num_batches_tracked else None
                modules.append(copy)
            elif isinstance(module, torch.nn.Flatten) or isinstance(module, onnx2pytorch.operations.flatten.Flatten):
                modules.append(diff_approx.Flatten(module.start_dim, module.end_dim))
            elif isinstance(module, torch.nn.Unflatten):
                modules.append(diff_approx.Unflatten(module.dim, module.unflattened_size))
            else:
                modules.append(module)
        return NeuralNetwork(modules=modules, **kwargs)

    def onnx_export(self, file: Any, disable_normalization=False, disable_unsqueeze=False, disable_squeeze=False,
                    input_sample: Union[torch.Tensor, str] = 'batch', **kwargs) -> None:
        """
        Saves this NeuralNetwork as a onnx formatted file.

        This method uses torch.onnx.export under the hood.
        It allows setting a few options that enable compatibility with ERAN.

        :param file: The file or file path to which the NeuralNetwork should be saved.
         All options supported by torch.onnx.export for argument 'f' are supported.
        :param disable_normalization: Exclude input and output normalisation from the the exported model.
         This also means that the mins, maxes, means and ranges tensors will not be contained in the onnx file.
         This option is mostly presented for compatibility with ERAN, that doesn't support the operations applied
         to normalize. Input normalisation values can be passed to ERAN separately.
        :param input_sample: Specify an input sample that will be used by
         torch.onnx.export to run the network and trace the computations. Alternatively, you may also specify two modes
         that will make this method create a sample of zeros for tracing.
         The two modes are specified as strings: 'batch', which will create a sample with the networks input shape
         and a single entry on the batch dimension and  'no batch', which will create a sample of zeros
         without a batch dimension.
         A custom input sample allows more control over the exported network. For example if you use an torch.int tensor
         as input, a cast operation to torch.float will also be included in the exported model, which will not be
         included if the input tensor has type torch.float. The default values will not export casts.
         Have a closer look at torch.onnx.export for more details on this parameter.
        :param kwargs: Further keyword arguments to torch.onnx.export
        """
        delegate = self

        # this class sets the normalize parameter to the value passed
        # to this method. This way this parameter is not included in the export ONNX model
        class ONNXExportNeuralNetworkWrapper(torch.nn.Sequential):
            def __init__(self):
                # explicitly do not call super().__init__
                # delegate is already initialised and all we want to do is relay calls to it
                pass

            def forward(self, inputs):
                return delegate.forward(inputs, normalize=not disable_normalization,
                                        disable_unsqueeze=disable_unsqueeze, disable_squeeze=disable_squeeze,
                                        disable_unflatten=True)

            def __getattr__(self, item):
                return getattr(delegate, item)

        wrapper = ONNXExportNeuralNetworkWrapper()

        assert not isinstance(input_sample, str) or input_sample == 'batch' or input_sample == 'no batch'
        if input_sample == 'batch':
            input_sample = torch.zeros((1, ) + self.inputs_shape, dtype=torch.float)
        elif input_sample == 'no batch':
            input_sample = torch.zeros(self.inputs_shape, dtype=torch.float)
        if input_sample.device != self.mins.device:
            input_sample = input_sample.to(self.mins.device)
        torch.onnx.export(wrapper, (input_sample, ), file, **kwargs)
