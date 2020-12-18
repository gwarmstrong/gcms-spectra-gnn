import json
import os

from gcms_spectra_gnn.data_models import MoleculeModel


class Backend:

    def __len__(self):
        raise NotImplementedError()

    def get_model(self, idx):
        raise NotImplementedError()

    def put_model(self, key, model):
        raise NotImplementedError()


class JSONDirectoryBackend(Backend):

    MAX_PER_DIR = 10000

    def __init__(self, library_path):
        # TODO could change the backend to take a dictionary (presumably
        #  parsed from JSON/yaml somewhere)
        with open(library_path) as fh:
            raw_library = json.load(fh)

        self.root_dir = os.path.dirname(library_path) + "/"
        library = [info for info in raw_library
                   if os.path.exists(self.root_dir + info['FP_PATH'])]
        # TODO how does the backend know if each entry has everything it needs
        #  to qualify to be used by a specific Dataset?
        self.library = [entry for entry in library if entry.get('FP_PATH')]
        self.model_cls = MoleculeModel

    def __len__(self):
        return len(self.library)

    def get_model(self, idx):
        model = self.model_cls.load(
            os.path.join(self.root_dir, self.library[idx]['FP_PATH'])
        )
        return model

    def put_model(self, key, model):
        # TODO this is currently not safe and will just overwrite whatever
        #  is at this path
        path = self._idxgen(key)
        model.save(path)

    def _idxgen(self, idx):
        base_path = os.path.join(self.root_dir,
                                 str(int(idx) // self.MAX_PER_DIR + 1)
                                 )
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        return os.path.join(base_path, idx + '.npz')
