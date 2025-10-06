from torch.utils.data import Dataset
import os

class exoplanet_dataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            root_path = os.path.join(root, 'k2')
        else:
            root_path = os.path.join(root, 'kepler')

        self.data = []
        self.labels = []
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            for file in os.listdir(label_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(label_path, file)
                    self.data.append(file_path)
                    self.labels.append(label)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    dataset = exoplanet_dataset(r'datas\lightcurves', train=True)
    print(len(dataset))
    print(dataset[0])