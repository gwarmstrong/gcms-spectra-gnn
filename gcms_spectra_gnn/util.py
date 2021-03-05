import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class SmilesException(Exception):
    pass


def read_smiles(smiles, add_h=True):
    """Reads a molecule from a SMILES string

    Args:
        add_h (bool): Adds hydrogens. Default: True

    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise SmilesException('MolFromSmiles Failed.')

    # Get the molecule from a canonical SMILES string
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, allHsExplicit=True))

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


def valid_elements(symbols, reference):
    """Tests a list for elements that are not in the reference.

    Args:
        symbols (list): The list whose elements to check.
        reference (list): The list containing all allowed elements.

    Returns:
        valid (bool): True if symbols only contains elements from the
        reference.

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
    num_at = mol.GetNumAtoms()
    indices = np.arange(num_at)
    np.random.shuffle(indices)
    indices = [int(i) for i in indices]

    # Renumber the atoms according to the random order
    new_mol = Chem.RenumberAtoms(mol, indices)

    return new_mol


def reorder_atoms(mol):
    """Reorders hydrogen atoms to appear following their heavy atoms in a
    molecule.

    Args:
        mol (Mol): A molecule (RDKit format)

    Returns:
        new_Mol (mol): A molecule with hydrogens after the corresponding heavy
        atoms.

    """

    # Create a list with old indices in new order
    indices = []
    for i, at in enumerate(mol.GetAtoms()):
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
    new_mol = Chem.RenumberAtoms(mol, indices)

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
        except:  # noqa: E722
            print('Failed to optimize conformer #%i.' % (i))

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

        xyz = np.empty([mol.GetNumAtoms(), 3])

        for ia, name in enumerate(symbols):

            position = conf.GetAtomPosition(ia)
            xyz[ia] = np.array([position.x, position.y, position.z])

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
    connect_matrix = np.zeros([num_at, num_at], dtype=int)

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            if bond is not None:
                connect_matrix[a.GetIdx(), b.GetIdx()] = 1

    return connect_matrix


def get_bonds_matrix(mol):
    """Provides bond types encoded as single (1.0). double (2.0),
    triple (3.0), and aromatic (1.5).

    Args:
        mol (Mol): a molecule in RDKit format

    Returns:
        connect_matrix (2D numpy array): connectivity matrix

    """

    # Initialization
    num_at = mol.GetNumAtoms()
    bonds_matrix = np.zeros([num_at, num_at])

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                bonds_matrix[a.GetIdx(), b.GetIdx()] = bt

    return bonds_matrix
