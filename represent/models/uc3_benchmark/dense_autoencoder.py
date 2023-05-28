import time
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class DenseEncoder(nn.Module):
    def __init__(self, input_shape: int, latent_dim: int):
        super().__init__()
        print(latent_dim)
        self.l1 = nn.Linear(in_features=input_shape, out_features=4 * latent_dim)
        self.l2 = nn.Linear(in_features=4 * latent_dim, out_features=2 * latent_dim)
        self.l3 = nn.Linear(in_features=2 * latent_dim, out_features=latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        latent = torch.relu(x)
        return latent


class DenseDecoder(nn.Module):
    def __init__(self, output_shape: int, latent_dim: int):
        super().__init__()
        self.l4 = nn.Linear(in_features=latent_dim, out_features=2 * latent_dim)
        self.l5 = nn.Linear(in_features=2 * latent_dim, out_features=4 * latent_dim)
        self.output = nn.Linear(in_features=4 * latent_dim, out_features=output_shape)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.l4(latent)
        x = torch.relu(x)
        x = self.l5(x)
        x = torch.relu(x)
        output = self.output(x)

        return output


class DenseAutoencoder(nn.Module):
    def __init__(self, input_shape, n_features, latent_dim):
        super().__init__()
        self.encoder = DenseEncoder(input_shape, latent_dim)
        self.decoder = DenseDecoder(input_shape, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, inputs):
        inputs=torch.squeeze(inputs)
        latent = self.encoder(inputs)
        output = self.decoder(latent)
        output = torch.unsqueeze(output,-1)
        return output
