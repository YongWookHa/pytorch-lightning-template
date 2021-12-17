from torch.utils.data import Dataset
from pathlib import Path


class DataModule(Dataset):
    def __init__(self, path):
        super(DataModule, self).__init__()
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class custom_collate(object):
    def __init__(self, ):
        pass

    def __call__(self, batch):
        """
        preprocess batch
        """
        return batch
