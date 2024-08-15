import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
