import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class SklearnTrainer:

    def fit(self, model, dataset):
        dataset.setup()
        training_data = dataset.train_dataset

        samples = [model_to_morgan_fingerprint(dataset[i])
                   for i in range(len(training_data))]

        X, y = dataset.collate_fn(samples)

        model.fit(X, y)


def model_to_morgan_fingerprint(model):
    N_BITS = 1024
    mol = Chem.MolFromSmiles(model.smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
       mol, 2, nBits=N_BITS,
    )
    arr = np.zeros(N_BITS, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def collate_arrays(samples):
    molecules, labels = map(list, zip(*samples))
    return torch.cat(molecules), torch.cat(labels)