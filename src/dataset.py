from torch.utils.data import Dataset

class exoplanet_dataset(Dataset):
    def __init__(self, root, train=True):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]