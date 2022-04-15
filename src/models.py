import torch
import torch.nn as nn

class loss_approximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        if torch.min(x)>0:
            input=torch.log(x)
            return torch.sigmoid(self.layers(input))
        else:
            return torch.ones((x.shape[0],1), requires_grad=True)

class pair_approximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8,4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        if torch.min(x)>0:
            input=torch.log10(x)
            return torch.sigmoid(self.layers(input))
        else:
            return torch.zeros((x.shape[0],1), requires_grad=True)



class lift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, dim, bias=False)
        )

    def forward(self):
        return torch.sigmoid(self.layers(torch.ones(1)))
