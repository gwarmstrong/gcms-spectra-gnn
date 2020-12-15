import os
import dgl
import json
import torch

from gcms_spectra_gnn.spectra_dataset import MoleculeModel
from gcms_spectra_gnn.molecule import ohe_molecules
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix


def collate_graphs(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(labels)


class MoleculeJSONDataset(Dataset):
    def __init__(self, library_path, graph_transform=None,
                 label_transform=None):

        # with open(os.path.join(library_path, 'index.json')) as fh:
        with open(library_path) as fh:
            library = json.load(fh)
        
        self.root_dir = os.path.dirname(library_path) + "/"
        library = [info for info in library
                   if os.path.exists(self.root_dir + info['FP_PATH'])]
        
        self.library = [entry for entry in library if entry.get('FP_PATH')]
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
