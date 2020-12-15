# Standard modules
from __future__ import print_function, division
import os
from io import StringIO
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# needed to make copies of data sets and variables
import copy

# Pytorch for data set
import torch
from torch.utils.data import Dataset, DataLoader

# RDkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# typing
from typing import List, Callable



######################
#  Helper functions  #
######################
class SmilesException(Exception):
    pass


def read_smiles(smiles,add_h=True):
    """Reads a molecule from a SMILES string

    Args:
        add_h (bool): Adds hydrogens. Default: True

    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise SmilesException('MolFromSmiles Failed.')

    # Get the molecule from a canonical SMILES string
    mol = Chem.MolFromSmiles( Chem.MolToSmiles(mol,allHsExplicit=True) )

    # Add hydrogens
    if add_h:
        mol = Chem.AddHs(mol)

    return mol


def safe_read_smiles(smiles, add_h):
    try:
        mol = read_smiles(smiles, add_h)
        return mol
    except SmilesException:
        return False


def valid_elements(symbols,reference):
    """Tests a list for elements that are not in the reference.

    Args:
        symbols (list): The list whose elements to check.
        reference (list): The list containing all allowed elements.

    Returns:
        valid (bool): True if symbols only contains elements from the reference.

    """

    valid = True

    if reference is not None:
        for sym in symbols:
            if sym not in reference:
                valid = False

    return valid


def reshuffle_atoms(mol):
    """Reshuffles atoms in a molecule.

    Args:
        mol (mol): A molecule (RDKit format)

    Returns:
        new_mol (mol): A molecule with shuffled atoms.

    """

    # Create an array with reshuffled indices
    num_at  = mol.GetNumAtoms()
    indices = np.arange(num_at)
    np.random.shuffle(indices)
    indices = [int(i) for i in indices]

    # Renumber the atoms according to the random order
    new_mol = Chem.RenumberAtoms(mol,indices)

    return new_mol



def reorder_atoms(mol):
    """Reorders hydrogen atoms to appear following their heavy atoms in a molecule.

    Args:
        mol (Mol): A molecule (RDKit format)

    Returns:
        new_Mol (mol): A molecule with hydrogens after the corresponding heavy atoms.

    """

    # Create a list with old indices in new order
    indices = []
    for i,at in enumerate(mol.GetAtoms()):
        # For each heavy atom
        if at.GetAtomicNum() != 1:
            # append its index
            indices.append(i)
            # search for its bonds
            for bond in at.GetBonds():
                end_idx = bond.GetEndAtomIdx()
                # check if the bond is to a hydrogen
                if mol.GetAtoms()[end_idx].GetAtomicNum() == 1:
                    # append the hydrogen's index (behind its heavy atom)
                    indices.append(end_idx)

    # Renumber the atoms according to the new order
    new_mol = Chem.RenumberAtoms(mol,indices)

    return new_mol


def generate_conformers(mol, n):
    """Generates multiple conformers per molecule.

    Args:
        mol (Mol): molecule (RDKit Mol format)
        n (int): number of conformers to generate.

    """

    indices = AllChem.EmbedMultipleConfs(mol, numConfs=n)

    for i in indices:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=i)
        except:
            print('Failed to optimize conformer #%i.'%(i))

    return


def get_coordinates_of_conformers(mol):
    """Reads the coordinates of the conformers.

    Args:
        mol (Mol): Molecule in RDKit format.

    Returns:
        all_conf_coord (list): Coordinates (one numpy array per conformer)

    """

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    all_conf_coord = []

    for ic, conf in enumerate(mol.GetConformers()):

        xyz = np.empty([mol.GetNumAtoms(),3])

        for ia, name in enumerate(symbols):

            position = conf.GetAtomPosition(ia)
            xyz[ia]  = np.array([position.x, position.y, position.z])

        all_conf_coord.append(xyz)

    return all_conf_coord


def get_connectivity_matrix(mol):
    """Generates the connection matrix from a molecule.

    Args:
        mol (Mol): a molecule in RDKit format

    Returns:
        connect_matrix (2D numpy array): connectivity matrix

    """

    # Initialization
    num_at = mol.GetNumAtoms()
    connect_matrix = np.zeros([num_at,num_at],dtype=int)

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx())
            if bond is not None:
                connect_matrix[a.GetIdx(),b.GetIdx()] = 1

    return connect_matrix


def get_bonds_matrix(mol):
    """Provides bond types encoded as single (1.0). double (2.0), triiple (3.0), and aromatic (1.5).

    Args:
        mol (Mol): a molecule in RDKit format

    Returns:
        connect_matrix (2D numpy array): connectivity matrix

    """

    # Initialization
    num_at = mol.GetNumAtoms()
    bonds_matrix = np.zeros([num_at,num_at])

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx())
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                bonds_matrix[a.GetIdx(),b.GetIdx()] = bt

    return bonds_matrix


class ModelError(ValueError):
    pass


class MoleculeModel:

    def __init__(self, smiles, num_at, symbols, at_nums, bonds, coords,
                 data, mol=None):
        # Append the SMILES
        self.smiles: str = smiles
        # Append the number of atoms
        self.num_at: int = num_at
        # Append all atom names and numbers
        self.symbols: List[str] = symbols
        self.at_nums: List[int] = at_nums
        # Append connectivity matrix and coordinates
        self.bonds: np.array = bonds
        self.coords: List[np.array] = coords
        # Append the values of the learned quantities
        # TODO this is where spectra should be stored
        self.data = data
        self.mol = mol

    def to_dict(self):
        return {
            'smiles': self.smiles,
            'num_at': self.num_at,
            'symbols': self.symbols,
            'at_nums': self.at_nums,
            'bonds': self.bonds,
            'coords': self.coords,
            'data': self.data,
        }

    def save(self, file):
        np.savez(file, **self.to_dict())

    @classmethod
    def from_raw_smiles(cls, raw_smiles, raw_data, add_h=True,
                        do_shuffle_atoms=False,
                        do_reorder_atoms=False, max_num_at=None,
                        max_num_heavy_at=None, elements=None,
                        num_conformers=1, bond_order=True,
                        ):
        # Reading raw data
        m = read_smiles(raw_smiles, add_h=add_h)

        # Shuffle the atom order
        if do_reorder_atoms:
            m = reorder_atoms(m)
        if do_shuffle_atoms:
            m = reshuffle_atoms(m)

        return cls.from_smiles(m, raw_data, max_num_at=max_num_at,
                               max_num_heavy_at=max_num_heavy_at,
                               elements=elements,
                               num_conformers=num_conformers,
                               bond_order=bond_order,
                               )

    @classmethod
    def from_smiles(cls, m, raw_data, max_num_at=None,
                    max_num_heavy_at=None, elements=None,
                    num_conformers=1, bond_order=True,
                    ):
        printer = print
        raw_num_at    = m.GetNumAtoms()
        raw_num_heavy = m.GetNumHeavyAtoms()
        # Read all atom names and numbers
        new_symbols = [a.GetSymbol() for a in m.GetAtoms()]
        new_at_nums = [a.GetAtomicNum() for a in m.GetAtoms()]

        # Check if the molecule is small enough
        small_enough = True
        if max_num_at is not None:
            if raw_num_at > max_num_at:
                small_enough = False
                printer('Too many atoms. Excluded from dataset.')
        if max_num_heavy_at is not None:
            if raw_num_heavy > max_num_heavy_at:
                small_enough = False
                printer('Too many heavy atoms. Excluded from dataset.')

        if small_enough:

            # Check for undesired elements
            if valid_elements(new_symbols, elements):

                # Track error messages
                Chem.WrapLogs()
                sio = sys.stderr = StringIO()
                # Generate the desired number of conformers
                generate_conformers(m, num_conformers)
                if 'ERROR' in sio.getvalue():
                    conf_coord = []
                    printer(sio.getvalue())
                else:
                    # Read the list of the coordinates of all conformers
                    conf_coord = get_coordinates_of_conformers(m)

                # only proceed if successfully generated conformers
                if len(conf_coord) > 0:
                    if bond_order:
                        # Generate the connectivity matrix with bond orders encoded as (1,1.5,2,3)
                        conmat = get_bonds_matrix(m)
                    else:
                        # Generate the connectivity matrix without bond orders
                        conmat = get_connectivity_matrix(m)

                    output_smiles = Chem.MolToSmiles(m, allHsExplicit=True)
                    model = cls(
                        smiles=output_smiles,
                        num_at=raw_num_at,
                        symbols=new_symbols,
                        at_nums=new_at_nums,
                        bonds=conmat,
                        coords=conf_coord,
                        data=raw_data,
                        mol=m,
                    )
                    return model
                else:
                    print('No conformers were generated. Excluded from dataset.')

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

    def add_noise(self, width, distribution='uniform'):
        """ Adds uniform or Gaussian noise to all coordinates.
            Coordinates are in nanometers.

        Args:
            width (float): The width of the distribution generating the noise.
            distribution(str): The distribution from with to draw. Either normal or uniform. Default: uniform.

        """
        # ... for each conformer ...
        for j, conf in enumerate(self.coords):
            # ... and for each atom
            for k, atom in enumerate(conf):
                if distribution == 'normal':
                    # add random numbers from a normal distribution.
                    self.coords[j][k] += np.random.normal(0.0,width,3)
                elif distribution == 'uniform':
                    # add random numbers from a uniform distribution.
                    self.coords[j][k] += width*(np.random.rand(3) - 0.5)


class Preprocessor:
    def __init__(self, name='molecules', alt_labels=None, elements=None,
                 add_h=True, order_atoms=False, shuffle_atoms=False,
                 num_conf=1, bond_order=True, max_num_at=None, max_num_heavy_at=None,
                 train_indices_raw=None, vali_indices_raw=None,
                 test_indices_raw=None,
                 # TODO directory and prefix
                 ):
        """Initializes a data set from a column in a CSV file.

        Args:
            name (str, opt.): Name of the dataset. Default: 'molecules'.
            alt_labels (list, opt.): Alternative labels for the properties, must be same length as col_names.
            elements (list, opt.): List of permitted elements (Element symbol as str). Default: all elements permitted.
            add_h (bool, opt.): Add hydrogens to the molecules. Default: True.
            order_atoms (bool, opt.): Atoms are ordered such that hydrogens directly follow their heavy atoms. Default: False.
            shuffle_atoms (bool, opt.): Atoms are randomly reshuffled (even if order_atoms is True). Default: False.

        """

        def none_to_list(arg):
            if arg is None:
                return []
            return arg

        self.add_h = add_h
        self.alt_labels = alt_labels
        self.elements = elements
        self.max_num_at = max_num_at
        self.max_num_heavy_at = max_num_heavy_at
        self.train_indices_raw = none_to_list(train_indices_raw)
        self.vali_indices_raw = none_to_list(vali_indices_raw)
        self.test_indices_raw = none_to_list(test_indices_raw)

        # Intitialize lists for filtered data
        self.smiles    = []
        self.num_at    = [] # number of atoms in each molecule
        self.symbols   = [] # lists of element symbols of each atom
        self.at_nums   = [] # lists of atomic numbers of each atom
        self.bonds     = []
        self.coords    = []
        self.data      = []
        self.mol       = []

        self.train_idx = []
        self.vali_idx  = []
        self.test_idx  = []

        # figure out which molecules were saved
        self.saved = []

        # Name of the dataset (some output options require it)
        self.name = name

        # Save properties
        self.atoms_ordered  = order_atoms
        self.atoms_shuffled = shuffle_atoms
        self.num_conformers = num_conf
        self.bond_order     = bond_order

    def transform(self, raw_data, raw_smiles, filenames=None):
        # Initialize new index
        new_index = 0

        # Reading raw data
        raw_mol       = [safe_read_smiles(s, add_h=self.add_h) for s in
                         raw_smiles]
        self.saved = []

        # For each molecule ...
        for im, m in enumerate(raw_mol):

            print('Processing '+str(im + 1)+'/'+str(len(raw_mol))+': '
                  ''+raw_smiles[im]+'.')
            if not raw_mol:
                self.saved.append(False)
                continue

            model = MoleculeModel.from_raw_smiles(
                raw_smiles=raw_smiles[im],
                raw_data=raw_data[im],
                add_h=self.add_h,
                do_shuffle_atoms=self.atoms_shuffled,
                do_reorder_atoms=self.atoms_ordered,
                max_num_at=self.max_num_at,
                max_num_heavy_at=self.max_num_heavy_at,
                elements=self.elements,
                num_conformers=self.num_conformers,
                bond_order=self.bond_order,
            )
            if model:
                # Append the molecule
                print('Added to the dataset.')
                if im in self.train_indices_raw:
                    self.train_idx.append(new_index)
                if im in self.vali_indices_raw:
                    self.vali_idx.append(new_index)
                if im in self.test_indices_raw:
                    self.test_idx.append(new_index)
                new_index += 1
                model.save(filenames[im])
                self.saved.append(True)
            else:
                self.saved.append(False)
                print('Contains undesired elements. Excluded from dataset.')



#######################
#  The Dataset Class  #
#######################
class MoleculesDataset(Dataset):
    """Dataset including coordinates and connectivity."""

    def __init__(self, raw_data, raw_smiles, name='molecules', alt_labels=None, elements=None,
                 add_h=True, order_atoms=False, shuffle_atoms=False,
                 num_conf=1, bond_order=True, max_num_at=None, max_num_heavy_at=None,
                 train_indices_raw=[], vali_indices_raw=[], test_indices_raw=[]):
        """Initializes a data set from a column in a CSV file.

        Args:
            csv_file (str): Path to the csv file with the data.
            col_names (str): Name of the columns with the properties to be trained.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
            alt_labels (list, opt.): Alternative labels for the properties, must be same length as col_names.
            elements (list, opt.): List of permitted elements (Element symbol as str). Default: all elements permitted.
            add_h (bool, opt.): Add hydrogens to the molecules. Default: True.
            order_atoms (bool, opt.): Atoms are ordered such that hydrogens directly follow their heavy atoms. Default: False.
            shuffle_atoms (bool, opt.): Atoms are randomly reshuffled (even if order_atoms is True). Default: False.

        """

        # Reading raw data
        raw_mol       = [read_smiles(s,add_h=add_h) for s in raw_smiles]
        raw_num_at    = [m.GetNumAtoms() for m in raw_mol]
        raw_num_heavy = [m.GetNumHeavyAtoms() for m in raw_mol]

        # Intitialize lists for filtered data
        self.smiles    = []
        self.num_at    = [] # number of atoms in each molecule
        self.symbols   = [] # lists of element symbols of each atom
        self.at_nums   = [] # lists of atomic numbers of each atom
        self.bonds     = []
        self.coords    = []
        self.data      = []
        self.mol       = []

        self.train_idx = []
        self.vali_idx  = []
        self.test_idx  = []

        # Name of the dataset (some output options require it)
        self.name = name

        # Save properties
        self.atoms_ordered  = order_atoms
        self.atoms_shuffled = shuffle_atoms
        self.num_conformers = num_conf
        self.bond_order     = bond_order

        # Initialize new index
        new_index = 0

        # For each molecule ...
        for im, m in enumerate(raw_mol):

            print('Processing '+str(im)+'/'+str(len(raw_mol))+': '+raw_smiles[im]+'.')

            # Shuffle the atom order
            if order_atoms:
                m = reorder_atoms(m)
            if shuffle_atoms:
                m = reshuffle_atoms(m)

            # Read all atom names and numbers
            new_symbols = [a.GetSymbol() for a in m.GetAtoms()]
            new_at_nums = [a.GetAtomicNum() for a in m.GetAtoms()]

            # Check if the molecule is small enough
            small_enough = True
            if max_num_at is not None:
                if raw_num_at[im] > max_num_at:
                    small_enough = False
                    print('Too many atoms. Excluded from dataset.')
            if max_num_heavy_at is not None:
                if raw_num_heavy[im] > max_num_heavy_at:
                    small_enough = False
                    print('Too many heavy atoms. Excluded from dataset.')

            if small_enough:

                # Check for undesired elements
                if valid_elements(new_symbols,elements):

                    # Track error messages
                    Chem.WrapLogs()
                    sio = sys.stderr = StringIO()
                    # Generate the desired number of conformers
                    generate_conformers(m, num_conf)
                    if 'ERROR' in sio.getvalue():
                        conf_coord = []
                        print(sio.getvalue())
                    else:
                        # Read the list of the coordinates of all conformers
                        conf_coord = get_coordinates_of_conformers(m)

                    # only proceed if successfully generated conformers
                    if len(conf_coord) > 0:
                        if self.bond_order:
                            # Generate the connectivity matrix with bond orders encoded as (1,1.5,2,3)
                            conmat = get_bonds_matrix(m)
                        else:
                            # Generate the connectivity matrix without bond orders
                            conmat = get_connectivity_matrix(m)
                        # Append the molecule
                        self.mol.append(m)
                        # Append the SMILES
                        self.smiles.append( raw_smiles[im] )
                        # Append the number of atoms
                        self.num_at.append( raw_num_at[im] )
                        # Append all atom names and numbers
                        self.symbols.append( new_symbols )
                        self.at_nums.append( new_at_nums )
                        # Append connectivity matrix and coordinates
                        self.bonds.append(conmat)
                        self.coords.append(conf_coord)
                        # Append the values of the learned quantities
                        self.data.append(raw_data[im])
                        print('Added to the dataset.')
                        if im in train_indices_raw:
                            self.train_idx.append(new_index)
                        if im in vali_indices_raw:
                            self.vali_idx.append(new_index)
                        if im in test_indices_raw:
                            self.test_idx.append(new_index)
                        new_index += 1
                    else:
                        print('No conformers were generated. Excluded from dataset.')
                else:
                    print('Contains undesired elements. Excluded from dataset.')


    def __len__(self):
        """Provides the number of molecules in a data set"""

        return len(self.smiles)


    def __getitem__(self, idx):
        """Provides a molecule from the data set.

        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.

        """

        sample = {'smiles': self.smiles[idx], \
                  'num_at': self.num_at[idx], \
                  'symbols': self.symbols[idx], \
                  'atomic numbers': self.at_nums[idx], \
                  'bonds': self.bonds[idx], \
                  'coords': self.coords[idx], \
                  'data': self.data[idx]}

        return sample

    def trim(self, idx):
        """Provides a molecule from the data set.

        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.

        """
        self.smiles = [self.smiles[i] for i in idx]
        self.num_at = [self.num_at[i] for i in idx]
        self.symbols = [self.symbols[i] for i in idx]
        self.at_nums = [self.at_nums[i] for i in idx]
        self.bonds = [self.bonds[i] for i in idx]
        self.coords = [self.coords[i] for i in idx]
        self.data = [self.data[i] for i in idx]


    def add_noise(self,width,distribution='uniform'):
        """ Adds uniform or Gaussian noise to all coordinates.
            Coordinates are in nanometers.

        Args:
            width (float): The width of the distribution generating the noise.
            distribution(str): The distribution from with to draw. Either normal or uniform. Default: uniform.

        """

        # For each molecule ...
        for i,mol in enumerate(self.coords):
            # ... for each conformer ...
            for j,conf in enumerate(mol):
                # ... and for each atom
                for k,atom in enumerate(conf):
                    if distribution == 'normal':
                        # add random numbers from a normal distribution.
                        self.coords[i][j][k] += np.random.normal(0.0,width,3)
                    elif distribution == 'uniform':
                        # add random numbers from a uniform distribution.
                        self.coords[i][j][k] += width*(np.random.rand(3) - 0.5)

        return


    def split(self,train_split=None,vali_split=0.1,test_split=0.1,shuffle=True,random_seed=None):
        """Creates data indices for training and validation splits.

        Args:
            vali_split (float): fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):  specifies random seed for shuffling. Default: None

        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.

        """

        dataset_size = len(self)
        indices = np.arange(dataset_size,dtype=int)

        # Calculate the numbers of elements per split
        vsplit = int(np.floor(vali_split * dataset_size))
        tsplit = int(np.floor(test_split * dataset_size))
        if train_split is not None:
            train = int(np.floor(train_split * dataset_size))
        else:
            train = dataset_size-vsplit-tsplit

        # Shuffle the dataset if desired
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)

        # Determine the indices of each split
        indices_test  = indices[:tsplit]
        indices_vali  = indices[tsplit:tsplit+vsplit]
        indices_train = indices[tsplit+vsplit:tsplit+vsplit+train]

        return indices_test, indices_vali, indices_train


    def element_statistics(self):
        """ Prints the numbers of molecules containing specific elements of the periodic table.
        """

        pte = Chem.GetPeriodicTable()

        el_names = [Chem.PeriodicTable.GetElementSymbol(pte,n) for n in range(1,113)]
        num_el_contained = np.zeros(len(el_names),dtype=int)

        for i,element in enumerate(el_names):
            el_count = np.array( [molsym.count(element) for molsym in self.symbols] )
            el_contained = el_count > 0
            num_el_contained[i] = np.sum(el_contained)

        sortidx = np.argsort(num_el_contained)
        sortidx = np.flip(sortidx)

        el_names = [el_names[i] for i in sortidx]
        num_el_contained = [num_el_contained[i] for i in sortidx]

        for line in np.array([el_names, num_el_contained]).T:
            if int(line[1]) > 0: print('%-2s %5i'%(line[0],int(line[1])))

        return el_names, num_el_contained


    def write_xyz(self,filename,prop_idx=0,indices=None):
        """Writes (a subset of) the data set as xyz file.

        Args:
            filename (str):  The name of the output file.
            prop_idx (int):  The index of the property to be trained for.
            indices (int[]): The indices of the molecules to write data for.

        """

        # Initialization
        if indices is None: indices = np.arange(len(self))

        with open(filename,'w') as out_file:

            # Header (only number of molecules)
            out_file.write(str(len(indices))+'\n')

            # For each molecule ...
            for i in indices:
                sample = self[i]
                # ... for  each conformer ...
                for pos in sample['coords']:
                    # write number of atoms
                    out_file.write(str(sample['num_at'])+'\n')
                    # write property to be trained for (now: only the first one)
                    out_file.write(str(sample['data'][prop_idx])+'\n')
                    # ... and for each atom:
                    for ia in range(sample['num_at']):
                        # write the coordinates.
                        out_file.write("%s %8.5f %8.5f %8.5f\n"%(sample['symbols'][ia], pos[ia,0], pos[ia,1], pos[ia,2]))

        return


    def write_connectivity_matrices(self,filename,prop_idx=0,indices=None,convert_atom_numbers=False):
        """Writes (a subset of) the data set as connectivity matrices.

        Args:
            filename (str):  The name of the output file.
            prop_idx (int):  The index of the property to be trained for.
            indices (int[]): The indices of the molecules to write data for.

        """

        # Initialization
        if indices is None: indices = np.arange(len(self))

        # Mapping of atom numbers for Risi's convention
        at_num_map = {1:2,6:1,7:4,8:3,9:5,15:6,16:7,17:8,5:9,35:10,53:11,14:12,34:13}

        with open(filename,'w') as out_file:

            # Header
            out_file.write('# ' + self.name+' '+self.labels[prop_idx]+'\n')
            out_file.write('\n'.join(self.labels)+'\n')
            out_file.write(str(len(indices))+'\n')

            # Molecule-specific information
            for i,idx in enumerate(indices):

                sample = self[idx]

                # numbers of atoms and value of the property
                out_file.write(str(sample['num_at']) + ' ' + str(sample['data'][prop_idx]) + '\n')

                # connectivity matrix
                for line in sample['bonds']:
                    pline = ' '.join( str(l) for l in line )
                    out_file.write(pline+'\n')

                # atomic numbers
                atomic_numbers = sample['atomic numbers']
                if convert_atom_numbers:
                    atomic_numbers = [at_num_map[a] for a in atomic_numbers]
                for an in atomic_numbers:
                    out_file.write(str(an)+'\n')


    def write_compressed(self,filename,prop_idx=0,indices=None):
        """Writes (a subset of) the data set as compressed numpy arrays.

        Args:
            filename (str):  The name of the output file.
            prop_idx (int):  The index of the property to be trained for.
            indices (int[]): The indices of the molecules to write data for.

        """

        # Define length of output
        if indices is None:
            indices = np.arange(len(self))

        # All arrays have the same size (the one of the biggest molecule)
        size = np.max( self.num_at )

        # Are bond orders encoded or not?
        if self.bond_order:
            bond_order_range = np.array([0,1,1.5,2,3])
        else:
            bond_order_range = np.array([0,1])

        # This is only called energies for traditional reasons.
        energies  = np.empty([len(indices)*self.num_conformers,1000])
        charges   = np.zeros([len(indices)*self.num_conformers,size])
        positions = np.zeros([len(indices)*self.num_conformers,size,3])
        bonds     = np.zeros([len(indices)*self.num_conformers,size,size,len(bond_order_range)],dtype=int)
        smiles    = []

        # For each molecule ...
        for i,idx in enumerate(indices):
            sample = self[idx]
            # ... for each conformer ...
            for ip,pos in enumerate(sample['coords']):
                # index for energies, charges, and positions
                # (such that in the output, there will first be one conformer of each molecule and then the next round of conformers)
                j = ip*len(indices)+i
                # add property to be trained
                energies[j] = sample['data'][prop_idx]
                # ... and for each atom:
                for ia in range(sample['num_at']):
                    charges[j,ia] = sample['atomic numbers'][ia]
                    positions[j,ia,0] = pos[ia,0]
                    positions[j,ia,1] = pos[ia,1]
                    positions[j,ia,2] = pos[ia,2]
                # add bonds. Bond order is one-hot encoded
                bonds_matrix  = sample['bonds']
                bonds_one_hot = (bond_order_range == bonds_matrix[...,None]).astype(int)
                bonds[j,:sample['num_at'],:sample['num_at'],:] = bonds_one_hot


        # Save as a compressed array
        np.savez_compressed(filename,spectra=energies,charges=charges,positions=positions,bonds=bonds)

        return
