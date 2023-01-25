# collection of useful functions to calculating
# the train and test set accuracies of networks
from tqdm import tqdm
import torch
from nn_repair.training.loss_functions import accuracy2


@torch.no_grad()
def full_accuracy(network, loader, progress_bar=True) -> float:
    data_iter = iter(loader)
    if progress_bar:
        data_iter = tqdm(data_iter)
    accuracy_sum = sum(accuracy2(targets, network(inputs)) for inputs, targets in data_iter)
    return accuracy_sum.item() / len(loader)


def get_stats(network, name, train_loader, test_loader):
    stats = {
        'network': name,
        'train_acc': full_accuracy(network, train_loader),
        'test_acc': full_accuracy(network, test_loader)
    }
    print(stats)
    return stats
