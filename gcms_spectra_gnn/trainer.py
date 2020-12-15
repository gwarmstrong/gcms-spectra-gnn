import torch
import torch.nn as nn
import pytorch_lightning as pl
from gcms_spectra_gnn.layers import Net
from gcms_spectra_gnn.dataset import MoleculeJSONDataset


DEFAULT_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']


class GCLightning(pl.LightningModule):

    def __init__(self, args):
        self.hparams = args
        # TODO: how to init??
        self.net = Net(len(DEFAULT_ELEMENTS),
                       length)

    def forward(self, smiles):
        """
        Parameters
        ----------
        smiles : list of str
           Smiles strings
        """
        # TODO: need to stub this out
        pass

    def train_dataloader(self):
        # TODO:
        return MoleculeJSONDataset()

    def val_dataloader(self):
        # TODO:
        return MoleculeJSONDataset()

    def training_step(self, batch, batch_idx):
        self.net.train()
        G, spec = batch
        # TODO: push data to GPU
        pred = self.net(G, G.ndata['mol_ohe'])
        loss = nn.MSELoss(pred, spec)
        # TODO: add more metrics as needed
        # Note that there is redundancy, which is OK
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        G, spec = batch
        # TODO: push data to GPU
        pred = self.net(G, G.ndata['mol_ohe'])
        loss = nn.MSELoss(pred, spec)
        # TODO: add more metrics as needed (i.e. AUPR, ...)
        tensorboard_logs = {'valid_loss': loss}
        return {'valid_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser)
        parser.add_argument(
            '--train-pairs', help='Training pairs file', required=True)
        parser.add_argument(
            '--test-pairs', help='Testing pairs file', required=True)
        parser.add_argument(
            '--valid-pairs', help='Validation pairs file', required=True)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        # TODO: hash out these args
        return parser
