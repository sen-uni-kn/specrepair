import torch


class Eval:
    """
    Puts a network or several networks into evaluation mode
    and restores the previous mode afterwards.
    """
    def __init__(self, *networks: torch.nn.Module):
        self.networks = networks
        self.training_modes = [network.training for network in networks]

    def __enter__(self):
        for network in self.networks:
            network.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for network, mode in zip(self.networks, self.training_modes):
            network.train(mode)


class Train:
    """
    Puts a network or several networks into training mode
    and restores the previous mode afterwards.
    """
    def __init__(self, *networks: torch.nn.Module):
        self.networks = networks
        self.training_modes = [network.training for network in networks]

    def __enter__(self):
        for network in self.networks:
            network.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for network, mode in zip(self.networks, self.training_modes):
            network.train(mode)
