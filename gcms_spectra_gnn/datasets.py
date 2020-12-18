from gcms_spectra_gnn.backend import Backend
from torch.utils.data import Dataset


class MoleculeJSONDataset(Dataset):
    def __init__(self, library: Backend, input_transform=None,
                 label_transform=None):

        self.library = library
        self.input_transform = input_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.library)

    def __getitem__(self, idx):
        model = self.library.get_model(idx)
        if self.input_transform:
            X = self.input_transform(model)
        else:
            X = model.to_safe_dict()
        if self.label_transform:
            y = self.label_transform(model)
        else:
            y = model.data

        return X, y
