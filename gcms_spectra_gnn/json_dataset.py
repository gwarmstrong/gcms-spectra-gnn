import os
import dgl

from gcms_spectra_gnn.molecule import MoleculeModel
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix


def basic_dgl_transform(molecule_model):
    """
    Transforms a molecular model into a dgl object

    TODO:  Change to a pytorch module.
    """
    G = dgl.from_scipy(coo_matrix(molecule_model.bonds))
    # TODO figure out some features!
    G.ndata['mol_ohe'] = ohe_molecules(molecule_model.symbols)
    return G


class MoleculeJSONDataset(Dataset):
    def __init__(self, library_path, graph_transform=None,
                 label_transform=None):

        with open(os.path.join(library_path, 'index.json')) as fh:
            library = json.load(fh)

        library = [info for info in library_info
                   if os.path.exists(library_path + info['FP_PATH'])]

        self.library = [entry for entry in library if entry.get('FP_PATH')]
        self.root_dir = os.path.dirname(library)
        self.graph_transform = graph_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.library)

    def __getitem__(self, idx):
        model = MoleculeModel.load(
            os.path.join(self.root_dir, self.library[idx]['FP_PATH']))
        if self.graph_transform:
            X = self.graph_transform(model)
        else:
            X = model.to_dict()
        if self.label_transform:
            y = self.label_transform(model)
        else:
            y = model.data

        return X, y
