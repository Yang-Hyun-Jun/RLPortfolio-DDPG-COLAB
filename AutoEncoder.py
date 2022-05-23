import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, state_dim=5):
        super().__init__()

        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 128)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, state_dim=5):
        super().__init__()

        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, state_dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.encoder = MLPEncoder(state_dim)
        self.decoder = MLPDecoder(state_dim)

    def forward(self, state):
        x = self.encoder(state)
        x = self.decoder(x)
        return x