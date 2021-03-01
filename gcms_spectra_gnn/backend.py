import json
import os
from pathlib import PurePath

from gcms_spectra_gnn.data_models import MoleculeModel, Model


class Backend:

    def __len__(self):
        raise NotImplementedError()

    def get_model(self, idx):
        raise NotImplementedError()

    def put_model(self, key, model):
        raise NotImplementedError()


class JSONDirectoryBackend(Backend):
    """Supports interacting with npz files indexed by a JSON for storing
    pre-processed data originating from an mgf


    Attributes
    ----------

    index_path : str
        Absolute path the the JSON file that provides an index for the dataset.
        The JSON should be comprised of an array of objects. All of the objects
        must have a 'SCANS' property.
    root_dir : str
        Absolute path to the directory that the dataset directory is in.
    library_all : dict of dict
        Contains all entries indexed in the library. Indexed by 'SCANS' value.
        Some entries may not have supporting data on the filesystem.
        Anything that might later be indexed must reside in this dictionary.
    library_can_get : dict of dict
        Indexes (by SCANS) all of the library entries that have a real
        filesystem counterpart and can be retrieved by this backend.
    int_index : list of dict
        Indexes, as a list, all of the library entries that have
        a real filesystem counterpart.

    """

    MAX_PER_DIR = 10000

    @staticmethod
    def index_loader(library_path):
        """Loads the library/index

        Parameters
        ----------
        library_path : str
            Path to the library JSON file

        Returns
        -------
        list of dict
            The library entries from the index.

        """
        with open(library_path) as fh:
            raw_library = json.load(fh)
        return raw_library

    @staticmethod
    def _abspath(path):
        return os.path.abspath(path)

    @staticmethod
    def _exists(path):
        return os.path.exists(path)

    @staticmethod
    def _pathgen(base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    model_cls = MoleculeModel

    def __init__(self, library_path):
        """

        Parameters
        ----------
        library_path : str
            Path to the JSON index

        """
        # TODO could change the backend to take a dictionary (presumably
        #  parsed from JSON/yaml somewhere)
        raw_library = self.index_loader(library_path)

        self.index_path = self._abspath(library_path)
        self.root_dir = os.path.dirname(self.index_path) + "/"
        # TODO how does the backend know if each entry has everything it needs
        #  to qualify to be used by a specific Dataset?
        #  maybe chain of responsibility? + bridge if working with multiple
        #  backends and have a way to have an abstraction to use in checks
        self.library_all = {int(entry['SCANS']): entry for entry in
                            raw_library}
        self.library_can_get = [v for v in self.library_all.values() if
                                v.get('FP_PATH') and self._exists(
                                 self.root_dir + v['FP_PATH'])
                                ]

    def __len__(self):
        # For use with dataloader, should only return the length of values
        # that have model data associated with them
        return len(self.library_can_get)

    def get_model(self, idx):
        """

        Parameters
        ----------
        idx : int
            Index of the model

        Returns
        -------
        model_cls
            A model object associated with the data at the index

        """
        model = self.model_cls.load(
            os.path.join(self.root_dir, self.library_can_get[idx]['FP_PATH'])
        )
        return model

    def collect(self, key):
        """Collect all values that contain a given key

        Parameters
        ----------
        key : int or str
            Key to retrieve the values (that exists) for

        Returns
        -------
        list of objects
            All values that have a given key.

        """
        return [entry[key] for entry in self.library_all.values() if key in
                entry]

    def put_model(self, key, model):
        """

        Parameters
        ----------
        key : int or str
            A key to store the model at
        model : Model
            The model to store with the given key

        Returns
        -------
        None

        """
        # TODO this is currently not safe and will just overwrite whatever
        #  is at this path
        path = self._idxgen(key)
        entry = self.library_all[key]
        entry['FP_PATH'] = str(PurePath(path).relative_to(self.root_dir))
        model.save(path)
        self.library_all.update({key: entry})
        self.library_can_get.append(entry)

    def flush(self):
        """Updates the on-disk index associated with the library.
        """
        return self._flusher(list(self.library_all.values()))

    def _flusher(self, values):
        with open(self.index_path, 'w') as fh:
            return json.dump(values, fh)

    def _idxgen(self, key):
        """Creates an on disk index corresponding to the key index

        Parameters
        ----------
        key : int
            The index of the data to generate a filesystem path for.

        Returns
        -------
        str
            The path where the record with idx should be stored.

        """
        base_path = os.path.join(self.root_dir,
                                 str(int(key) // self.MAX_PER_DIR + 1)
                                 )
        self._pathgen(base_path)
        return os.path.join(base_path, str(key) + '.npz')
