import numpy as np
import torch

DEFAULT_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']


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
    return torch.tensor(ohe)
