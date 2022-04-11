import torch

class EValDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.data = X


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def make_dataloader(evals, batch_size):
    return torch.utils.data.DataLoader(EValDataset(evals), batch_size=batch_size, shuffle=True)
