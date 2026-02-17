import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=9):
        # Super fait en sorte que la classe Autoencoder peut
        # utiliser les méthodes de la classe fondatrice Module
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

        self.decoder = nn.Sequential(
            nn.Linear(6, 7),
            nn.ReLU(),
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat