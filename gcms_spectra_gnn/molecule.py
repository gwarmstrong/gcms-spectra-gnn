import numpy as np
import torch
import dgl
from scipy.sparse import coo_matrix

DEFAULT_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br',
                    'I', 'P', 'Si']


def basic_dgl_transform(model):
    G = dgl.from_scipy(coo_matrix(model.bonds))
    # TODO figure out some features!
    G.ndata['mol_ohe'] = ohe_molecules(model.symbols)
    return G
        

def ohe_molecules(symbols, elements=None):
    # TODO this might error if we get something not in elements
    # Be more explicit about error check
    if elements is None:
        elements = DEFAULT_ELEMENTS
    positions = np.zeros(len(symbols)).astype(int)
    for i, atom in enumerate(symbols):
        positions[i] = elements.index(atom)
    ohe = np.zeros((len(symbols), len(elements)))
    ohe[np.arange(len(positions)), positions] = 1
    return torch.tensor(ohe).float()


class OneHotSpectrumEncoder:

    def __init__(self, min_=20, max_=500):
        """
        Parameters
        ----------
        min_ : int
            Lowest allowed m/z. Values lower are excluded.
        max_ : int
            Highest allowed m/z. Values higher are excluded.

        """
        self.min_ = min_
        self.max_ = max_
        self.width = max_ - min_ + 1

    def __call__(self, model):
        """
        Parameters
        ----------

        model : MoleculeModel

        Returns
        -------
        torch.tensor
            One hot encoding of the molecules atoms

        """
        y = np.zeros([1, self.width])
        for i, idx in enumerate(model.data[0, :]):
            idx = int(idx)
            if (idx >= self.min_) and (idx <= self.max_):
                y[0, idx - self.min_] = model.data[1, i]
        return torch.tensor(y)
