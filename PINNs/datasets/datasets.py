from torch.utils.data import Dataset, DataLoader


class BounDataset(Dataset):
    def __init__(self, x, y):
        super(DataSetBoundary, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, item):
        x = self.x[item, :]
        y = self.y[item, :]
        return x, y

    def __len__(self):
        return self.x.shape[0]
