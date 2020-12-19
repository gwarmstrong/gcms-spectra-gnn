from gcms_spectra_gnn.data_models import MoleculeModel
from gcms_spectra_gnn.util import safe_read_smiles


class Preprocessor:
    def __init__(self,
                 backend,
                 elements=None,
                 add_h=True,
                 ):
        """Initializes a data set from a column in a CSV file.

        """
        self.backend = backend
        self.add_h = add_h
        self.elements = elements

        # Intitialize lists for filtered data
        self.mol = []

        # figure out which molecules were saved
        self.saved = []

    def transform(self, raw_data, raw_smiles, indices, hook):
        # Initialize new index
        # Reading raw data
        raw_mol = [safe_read_smiles(s, add_h=self.add_h) for s in
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
                elements=self.elements,
            )
            if model:
                # Append the molecule
                print('Added to the dataset.')
                self.backend.put_model(indices[im], model)
                self.saved.append(True)
            else:
                self.saved.append(False)
                print('Contains undesired elements. Excluded from dataset.')

            if hook is not None:
                hook(im, self)
