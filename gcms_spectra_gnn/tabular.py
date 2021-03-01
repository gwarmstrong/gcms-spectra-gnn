import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class SklearnTrainer:

    def fit(self, model, dataset):
        dataset.setup()
        training_data = dataset.train_dataset

        samples = [training_data[i] for i in range(len(training_data))]

        X, y = dataset.collate_fn(samples)

        model.fit(X, y)


def model_to_morgan_fingerprint(model, n_bits=1024):
    mol = Chem.MolFromSmiles(model.smiles)
    arr = _fingerprint(mol, n_bits)
    return torch.tensor(arr).reshape(1, -1).float()


def mol_to_morgan_fingerprint(mol, n_bits=1024):
    arr = _fingerprint(mol, n_bits)
    return arr.tolist()


def _fingerprint(mol, n_bits):
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=n_bits,
    )
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def collate_arrays(samples):
    molecules, labels = map(list, zip(*samples))
    return torch.cat(molecules), torch.cat(labels)
