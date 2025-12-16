import torch
import torch.nn as nn

class ResidualGenerator(nn.Module):
    def __init__(self, gene_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(gene_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, gene_dim)
        self.act = nn.ReLU()

    def forward(self, x, drug, cell):
        inp = torch.cat([x, drug, cell], dim=1)
        h = self.act(self.fc1(inp))
        h = self.act(self.fc2(h))
        return x + self.fc3(h)


class MLPDiscriminator(nn.Module):
    def __init__(self, gene_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(gene_dim + 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, drug, cell):
        inp = torch.cat([x, drug, cell], dim=1)
        return self.model(inp)
