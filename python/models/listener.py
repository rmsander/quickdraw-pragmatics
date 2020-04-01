"""PyTorch neural network model for the listener agent.  This network
consists of a 1D CNN with max pooling to capture semantic features of from
the sequences of 2D points given to the listener, as well as adhere to
permutation invariance of the input sequences."""

# External package imports
import torch.nn as nn


# TODO: Insert dimensions
class ListeNet(nn.Module):
    """Neural network object representing our listener network.  This network
    takes as input a sequence of 2D points corresponding to sampled
    components of the sketch uttered by the listener."""

    # Constructor method for class
    def __init__(self, in_channels=10, K_conv=5, K_mp=5, fc_neurons=500,
                 num_classes=10):
        # Inherit from superclass
        super(ListeNet, self).__init__()

        # Define parameters of network
        self.in_channels = in_channels
        self.K_conv = K_conv
        self.K_mp = K_mp

        # Feature detection
        self.conv1 = nn.Conv1d(self.in_channels, 1, kernel_size=self.K_conv)
        self.maxpool1 = nn.MaxPool1d(self.K_mp)

        # Multilayer Perceptron
        self.fc1 = nn.Linear(fc_neurons, fc_neurons // 4)
        self.fc2 = nn.Linear(fc_neurons // 4, fc_neurons // 16)

        # Classification layer
        self.SM = nn.Softmax(dim=num_classes)

    # Forward method for mapping input to output
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.maxpool1(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.SM(x)
