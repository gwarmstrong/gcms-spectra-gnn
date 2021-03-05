import os
import json
from unittest import TestCase

from gcms_spectra_gnn.data_models import Model
from gcms_spectra_gnn.backend import JSONDirectoryBackend


class MockModel(Model):

    def __init__(self, data=None):
        self.data = data
        self.saved = False

    @classmethod
    def load(cls, path):
        return cls(path)

    def save(self, path):
        self.saved = path


class TestJSONDirectoryBackend(TestCase):

    def setUp(self):
        super().setUp()
        self.index_loader_backup = JSONDirectoryBackend.index_loader

        def mock_index(*args, **kwargs):
            return [
                {
                    'SCANS': '1',
                    'FP_PATH': '1/1.npz',
                },
                {
                    'SCANS': '2',
                    'FP_PATH': '1/2.npz',
                },
                {
                    'SCANS': '10001',
                    'FP_PATH': '2/10001.npz',
                },
                {
                    'SCANS': '10002',
                },
            ]

        JSONDirectoryBackend.index_loader = mock_index

        @staticmethod
        def mock_path(path):
            return os.path.join('/some/home/dir', path)

        @staticmethod
        def mock_exists(path):
            return True

        @staticmethod
        def mock_flusher(data):
            return json.dumps(data)

        @staticmethod
        def mock_pathgen(path):
            pass

        self.abspath_backup = JSONDirectoryBackend._abspath
        self.exists_backup = JSONDirectoryBackend._exists
        self.model_backup = JSONDirectoryBackend.model_cls
        self.flusher_backup = JSONDirectoryBackend._flusher
        self.pathgen_backup = JSONDirectoryBackend._pathgen
        JSONDirectoryBackend._abspath = mock_path
        JSONDirectoryBackend._exists = mock_exists
        JSONDirectoryBackend.model_cls = MockModel
        JSONDirectoryBackend._flusher = mock_flusher
        JSONDirectoryBackend._pathgen = mock_pathgen

    def tearDown(self):
        JSONDirectoryBackend.index_loader = self.index_loader_backup
        JSONDirectoryBackend._abspath = self.abspath_backup
        JSONDirectoryBackend._exists = self.exists_backup
        JSONDirectoryBackend.model_cls = self.model_backup
        JSONDirectoryBackend._flusher = self.flusher_backup
        JSONDirectoryBackend._pathgen = self.pathgen_backup

    def test_collect(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        obs_scans = backend.collect('SCANS')
        exp_scans = ['1', '2', '10001', '10002']
        self.assertCountEqual(obs_scans, exp_scans)

        obs_path = backend.collect('FP_PATH')
        exp_path = ['1/1.npz', '1/2.npz', '2/10001.npz']
        self.assertCountEqual(obs_path, exp_path)

    def test_init(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        obs_index_path = backend.index_path
        exp_index_path = '/some/home/dir/mock/library/path/to.json'
        self.assertEqual(exp_index_path, obs_index_path)
        obs_root_dir = backend.root_dir
        exp_root_dir = '/some/home/dir/mock/library/path/'
        self.assertEqual(exp_root_dir, obs_root_dir)

    def test_len(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        exp_len = 3
        obs_len = len(backend)
        self.assertEqual(exp_len, obs_len)

    def test_get_model(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        model = backend.get_model(0)
        obs = model.data
        exp = '/some/home/dir/mock/library/path/1/1.npz'
        self.assertEqual(exp, obs)

    def test_put_model(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        model = MockModel()
        test_idx = 10002
        self.assertNotIn('FP_PATH', backend.library_all[test_idx])
        self.assertFalse(any(entry['SCANS'] == '10002'
                             for entry in backend.library_can_get))
        self.assertNotIn(test_idx, backend.library_can_get)
        backend.put_model(test_idx, model)
        exp_saved = '/some/home/dir/mock/library/path/2/10002.npz'
        obs_saved = model.saved
        self.assertEqual(exp_saved, obs_saved)
        self.assertIn('FP_PATH', backend.library_all[test_idx])
        self.assertTrue(any(entry['SCANS'] == '10002'
                            for entry in backend.library_can_get))
        obs_library_entry = [entry for entry in backend.library_can_get
                             if entry['SCANS'] == str(test_idx)
                             ][0]
        exp_library_entry = {
            'SCANS': '10002',
            'FP_PATH': '2/10002.npz',
        }
        self.assertDictEqual(exp_library_entry, obs_library_entry)

    def test_flush(self):
        library_path = 'mock/library/path/to.json'
        backend = JSONDirectoryBackend(library_path)
        model = MockModel()
        test_idx = 10002
        backend.put_model(test_idx, model)
        flushed_result = backend.flush()
        obs_result = json.loads(flushed_result)
        exp_result = \
            [
                {
                    'SCANS': '1',
                    'FP_PATH': '1/1.npz',
                },
                {
                    'SCANS': '2',
                    'FP_PATH': '1/2.npz',
                },
                {
                    'SCANS': '10001',
                    'FP_PATH': '2/10001.npz',
                },
                {
                    'SCANS': '10002',
                    'FP_PATH': '2/10002.npz',
                },
            ]
        self.assertCountEqual(exp_result, obs_result)
