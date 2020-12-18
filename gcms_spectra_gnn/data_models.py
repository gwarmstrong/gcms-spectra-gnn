from typing import List, Callable
import numpy as np
from rdkit import Chem

from gcms_spectra_gnn.util import read_smiles, valid_elements, get_bonds_matrix


class ModelError(ValueError):
    pass


class MoleculeModel:

    def __init__(self,
                 smiles=None,
                 explicit_smiles=None,
                 num_at=None,
                 symbols=None,
                 at_nums=None,
                 bonds=None,
                 data=None,
                 mol=None,
                 **kwargs
                 ):
        # Append the SMILES
        self.smiles: str = smiles
        self.explicit_smiles = explicit_smiles
        # Append the number of atoms
        self.num_at: int = num_at
        # Append all atom names and numbers
        self.symbols: List[str] = symbols
        self.at_nums: List[int] = at_nums
        # Append connectivity matrix and coordinates
        self.bonds: np.array = bonds
        # Append the values of the learned quantities
        # TODO this is where spectra should be stored
        self.data = data
        self.mol = mol

    def to_safe_dict(self):
        return {
            'smiles': self.smiles,
            'explicit_smiles': self.explicit_smiles,
            'num_at': self.num_at,
            'symbols': self.symbols,
            'at_nums': self.at_nums,
            'bonds': self.bonds,
            'data': self.data,
        }

    def save(self, file):
        np.savez(file, **self.to_safe_dict())

    @classmethod
    def from_raw_smiles(cls, raw_smiles, raw_data, add_h=True,
                        elements=None,
                        ):
        # Reading raw data
        m = read_smiles(raw_smiles, add_h=add_h)
        return cls.from_mol(m, raw_data,
                            elements=elements,
                            )

    @classmethod
    def from_mol(cls,
                 m,
                 raw_data,
                 elements=None,
                 ):
        raw_num_at = m.GetNumAtoms()
        # Read all atom names and numbers
        new_symbols = [a.GetSymbol() for a in m.GetAtoms()]
        new_at_nums = [a.GetAtomicNum() for a in m.GetAtoms()]

        # Check for undesired elements
        if valid_elements(new_symbols, elements):
            conmat = get_bonds_matrix(m)

            output_smiles = Chem.MolToSmiles(m, allHsExplicit=True)
            model = cls(
                smiles=output_smiles,
                num_at=raw_num_at,
                symbols=new_symbols,
                at_nums=new_at_nums,
                bonds=conmat,
                data=raw_data,
                mol=m,
            )
            return model

        return False

    @classmethod
    def load(cls, file, data_fn: Callable = None, allow_pickle: bool = False):
        if data_fn is None:
            def data_fn(data):
                return data

        npzfile = np.load(file, allow_pickle=allow_pickle)
        return cls(
            npzfile['smiles'].item(),
            npzfile['num_at'].item(),
            list(npzfile['symbols']),
            list(npzfile['at_nums']),
            npzfile['bonds'],
            npzfile['coords'],
            data_fn(npzfile['data']),
        )
