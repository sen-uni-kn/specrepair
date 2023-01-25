# downloads and convers the MNIST model used by
# Ben Goldberger, Guy Katz, Yossi Adi, Joseph Keshet:
# Minimal Modifications of Deep Neural Networks using Verification. LPAR 2020: 260-278
# The network is available online, but without licensing information, so
# it can not be included in this repository

import requests
from pathlib import Path
from tempfile import TemporaryDirectory

from tensorflow.keras.models import model_from_json

import torch
from torch.utils.data import DataLoader
from deep_opt import NeuralNetwork
import deep_opt.models.differentiably_approximatable_nn_modules as nn_layers
from nn_repair.training.loss_functions import accuracy2

from torchvision import datasets, transforms

if __name__ == '__main__':
    json_file_url = 'https://github.com/jjgold012/MinimalDNNModificationLpar2020/raw/' \
                    '1c967dbdbf2dc7facafbdbb03c96f0bce6f3b78b/WatermarkRemoval/Models/mnist.w.wm_model.json'
    md5_file_url = 'https://github.com/jjgold012/MinimalDNNModificationLpar2020/raw/' \
                   '1c967dbdbf2dc7facafbdbb03c96f0bce6f3b78b/WatermarkRemoval/Models/mnist.w.wm_model.h5'
    print("Downloading files...")
    with TemporaryDirectory() as temp_dir:
        json_remote = requests.get(json_file_url)
        h5_remote = requests.get(md5_file_url)
        json_path = Path(temp_dir, 'model.json')
        h5_path = Path(temp_dir, 'model.h5')
        with open(json_path, 'wb') as json_file:
            json_file.write(json_remote.content)
        with open(h5_path, 'wb') as h5_file:
            h5_file.write(h5_remote.content)

        with open(json_path, 'rt') as json_file:
            with open(h5_path, 'rb') as h5_file:
                loaded_model_json = json_file.read()
                print(loaded_model_json)
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(h5_file.name)
    print("Model loaded.")
    net_copy = NeuralNetwork(
        means_inputs=0, ranges_inputs=1/255,  # the downloaded network expects inputs in [0, 255], not [0, 1]
        inputs_shape=(1, 28, 28),
        modules=[
            nn_layers.Flatten(),
            nn_layers.Linear(784, 150), nn_layers.ReLU(),
            nn_layers.Linear(150, 10, bias=False)
        ]
    )
    # copying over weights
    net_copy[1].weight.data = torch.tensor(loaded_model.layers[2].kernel.numpy()).float().T
    net_copy[1].bias.data = torch.tensor(loaded_model.layers[2].bias.numpy()).float()
    net_copy[3].weight.data = torch.tensor(loaded_model.layers[3].kernel.numpy()).float().T
    # last layer of loaded model does not have a bias

    print("Converted Model. Testing...")
    mnist_test_set = datasets.MNIST('../../datasets', train=False, transform=transforms.ToTensor())
    inputs, targets = next(iter(DataLoader(mnist_test_set, batch_size=len(mnist_test_set))))
    accuracy = accuracy2(targets, net_copy(inputs))
    print(f"Test set accuracy of converted model: {accuracy*100:.0f}%. Should be: 97%")
    assert 0.9680 < accuracy <= 0.97
    output_file = 'goldberger_et_al_mnist_watermark_model.pyt'
    print(f"Saving converted model in file: {output_file}")
    torch.save(net_copy, output_file)

