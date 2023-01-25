import torch
from deep_opt import NeuralNetwork
import deep_opt.models.differentiably_approximatable_nn_modules as nn_layers

import numpy as np
import pandas

import matplotlib.pyplot as plt


if __name__ == "__main__":
    DATASET_FILE = '../resources/collision_detection/dataset.csv'
    OUTPUT_FILE = '../resources/collision_detection/CollisionDetection_test_network'

    print('Loading data set...')
    # 1 means collision in the dataset
    dataset = pandas.read_csv('../resources/collision_detection/dataset.csv', header=None,
                              names=['x', 'y', 's', 'd', 'c1', 'c2', 'class'])
    num_inputs = 6
    # first four variables are in [0,1], last two are in [-1, 1]
    norm_mins = [0] * 4 + [-1] * 2
    norm_maxes = [1] * 6
    norm_means_inputs = [0] * 6  # mean=0, range=1 disables the mean/range normalisation
    norm_ranges_inputs = [1] * 6
    norm_means_outputs = 0  # use same normalisation constant for all outputs
    norm_ranges_outputs = 1

    print(f'n: {len(dataset)}')
    print(dataset.iloc[0:5])
    print()
    print("Class frequency")
    print(dataset['class'].value_counts())

    # make a train/test split
    rand_state = np.random.RandomState(seed=2103)
    train_data = dataset.sample(frac=2/3, random_state=rand_state)
    test_data = dataset.drop(index=train_data.index)
    train_data_X = torch.as_tensor(train_data.drop('class', axis=1).to_numpy())
    train_data_y = torch.as_tensor(train_data['class'].to_numpy())
    test_data_X = torch.as_tensor(test_data.drop('class', axis=1).to_numpy())
    test_data_y = torch.as_tensor(test_data['class'].to_numpy())
    print()
    print(f'Created train/test split: {len(train_data)} samples for training; {len(test_data)} for testing.')

    train_data.to_csv(OUTPUT_FILE + '_train_data.csv', index=False)
    test_data.to_csv(OUTPUT_FILE + '_test_data.csv', index=False)

    # More information on configurations in README file
    # config: 1 (ReLU 1)
    # layers = [
    #     nn_layers.Linear(in_features=6, out_features=10),
    #     nn_layers.ReLU(),
    #     nn_layers.Linear(in_features=10, out_features=10),
    #     nn_layers.ReLU(),
    #     nn_layers.Linear(in_features=10, out_features=2),  # two classes
    # ]
    # config: 2 (tanh 1)
    # layers = [
    #     nn_layers.Linear(in_features=6, out_features=6),
    #     nn_layers.Tanh(),
    #     nn_layers.Linear(in_features=6, out_features=2),  # two classes
    # ]
    # config: 3 (tanh 2)
    # layers = [
    #     nn_layers.Linear(in_features=6, out_features=10),
    #     nn_layers.Tanh(),
    #     nn_layers.Linear(in_features=10, out_features=10),
    #     nn_layers.Tanh(),
    #     nn_layers.Linear(in_features=10, out_features=2),  # two classes
    # ]
    # config: 4 (MaxPool 1)
    # layers = [
    #     nn_layers.Linear(in_features=6, out_features=40),
    #     nn_layers.Unflatten(1, (1, 40)),  # add a channel dimension for MaxPool
    #     nn_layers.MaxPool1d(kernel_size=(4,)),
    #     nn_layers.Flatten(),  # remove the channel dimension again
    #     nn_layers.Linear(in_features=10, out_features=19),
    #     nn_layers.ReLU(),
    #     nn_layers.Linear(in_features=19, out_features=2),
    #     nn_layers.ReLU()
    # ]
    # config: 5 (ReLU 2)
    # layers = [
    #     nn_layers.Linear(in_features=6, out_features=100),
    #     nn_layers.ReLU(),
    #     nn_layers.Linear(in_features=100, out_features=10),
    #     nn_layers.ReLU(),
    #     nn_layers.Linear(in_features=10, out_features=2),  # two classes
    # ]
    #
    # network = NeuralNetwork(
    #     mins=norm_mins, maxes=norm_maxes,
    #     means_inputs=norm_means_inputs, ranges_inputs=norm_ranges_inputs,
    #     means_outputs=norm_means_outputs, ranges_outputs=norm_ranges_outputs,
    #     modules=layers
    # )

    # config: 6 (ReLU 3)
    layers = [
        nn_layers.Linear(in_features=6, out_features=60),
        nn_layers.ReLU(),
        nn_layers.Linear(in_features=60, out_features=60),
        nn_layers.ReLU(),
        nn_layers.Linear(in_features=60, out_features=60),
        nn_layers.ReLU(),
        nn_layers.Linear(in_features=60, out_features=2),  # two classes
    ]

    network = NeuralNetwork(
        mins=norm_mins, maxes=norm_maxes,
        means_inputs=norm_means_inputs, ranges_inputs=norm_ranges_inputs,
        means_outputs=norm_means_outputs, ranges_outputs=norm_ranges_outputs,
        modules=layers
    )

    loss_function = torch.nn.CrossEntropyLoss()
    # config 1-3
    # optim = torch.optim.Adam(network.parameters(), lr=0.01)
    # scheduler = None
    # config 4
    # optim = torch.optim.Adam(network.parameters(), lr=0.005)
    # scheduler = None
    # config 5
    # optim = torch.optim.Adam(network.parameters(), lr=0.005, weight_decay=0.0025)
    # scheduler = None
    # config 6
    optim = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.0025)
    scheduler = None

    print('Training...')

    def calc_loss():
        if torch.is_grad_enabled():
            optim.zero_grad()
        pred = network(train_data_X)
        loss = loss_function(pred, train_data_y)
        if loss.requires_grad:
            loss.backward()
        return loss


    # config 1+3:
    # max_epochs = 1000
    # config 2:
    # max_epochs = 2500
    # config 4:
    # max_epochs = 2000
    # config 5+6:
    max_epochs = 5000

    train_loss = []
    test_loss = []
    for i in range(max_epochs):
        optim.step(calc_loss)
        if scheduler is not None:
            scheduler.step()

        train_loss.append(calc_loss())
        # calculate test loss
        prediction = network(test_data_X)
        test_loss.append(loss_function(prediction, test_data_y))
        if i % 100 == 0:
            accuracy = (torch.argmax(network(train_data_X), dim=1) == train_data_y).float().mean() * 100
            print(f'Iteration: {i}.\n'
                  f'Training accuracy: {accuracy:.2f}%\n'
                  f'Training loss: {train_loss[-1]:.4}\nTest loss: {test_loss[-1]:.4}')

    plt.plot(train_loss, label='training loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.show()

    # save the model
    torch.save(network, OUTPUT_FILE + '.pyt')
