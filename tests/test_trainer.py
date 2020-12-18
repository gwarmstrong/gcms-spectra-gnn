import pytest
from gcms_spectra_gnn.trainer import GCLightning


class TestTrainer(object):
    @pytest.mark.parametrize('in_feats', [3, 5])
    @pytest.mark.parametrize('out_feats', [2, 4])
    def test_initialization(self, in_feats, out_feats):
        model_args = {"input_features": in_feats, "output_features": out_feats}
        training_args = {}
        GCLightning(training_args, model_args)
