import torch
from torch import nn
import pytorch_lightning as pl
import argparse
from gcms_spectra_gnn.models import Net
from gcms_spectra_gnn.datasets import MoleculeJSONDataset
from gcms_spectra_gnn.backend import JSONDirectoryBackend
from gcms_spectra_gnn.transforms import collate_graphs
from gcms_spectra_gnn.transforms import (
    basic_dgl_transform, OneHotSpectrumEncoder)
from torch.utils.data import DataLoader


def mse_loss(pred, spec):
    return ((pred - spec)**2).mean()


def tag(log_dict, tag):
    return {'/'.join([tag, key]): value for key, value in log_dict.items()}


class MeanCosineSimilarity:

    def __init__(self, round_preds=False, cosine_kwargs=None):
        self.round = round_preds
        if cosine_kwargs is None:
            cosine_kwargs = dict()
        self.cos = nn.CosineSimilarity(**cosine_kwargs)

    def __call__(self, pred, label):
        if self.round:
            pred = torch.round(pred)
        return self.cos(pred, label).mean()


class MoleculeJSONDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MoleculeJSONDataModule, self).__init__()
        self.input_transform = basic_dgl_transform
        self.label_transform = OneHotSpectrumEncoder()
        self.collate_fn = collate_graphs
        self.hparams = args

    def prepare_data(self, *args, **kwargs):
        # TODO if needed
        pass

    def setup(self, stage=None):
        library = JSONDirectoryBackend(self.hparams.train_library)
        self.train_dataset = MoleculeJSONDataset(
            library,
            input_transform=self.input_transform,
            label_transform=self.label_transform,
        )
        print("train dataset length:", len(self.train_dataset))
        library = JSONDirectoryBackend(self.hparams.valid_library)
        self.valid_dataset = MoleculeJSONDataset(
            library,
            input_transform=self.input_transform,
            label_transform=self.label_transform,
        )
        print("valid dataset length:", len(self.valid_dataset))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(
            self.valid_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return valid_dataloader


class GCLightning(pl.LightningModule):

    def __init__(self, args, model_init_args):
        super(GCLightning, self).__init__()
        """
        Constructs the trainer

        Parameters
        ----------
        args : dict
            Training arguments
        model_init_args : dict
            Arguments for initializing the model.
        """
        self.hparams = args
        # TODO: how to init??
        self.net = Net(**model_init_args)
        self.loss_fn = mse_loss
        self.eval_metrics = [
            ('cosine', MeanCosineSimilarity()),
            ('rounded_cosine', MeanCosineSimilarity(round_preds=True))
        ]

    def _calc_eval_metrics(self, pred, label):
        return {key: fn(pred, label) for key, fn in self.eval_metrics}

    def forward(self, smiles):
        """
        Parameters
        ----------
        smiles : list of str
           Smiles strings
        """
        # TODO: need to stub this out
        pass

    def training_step(self, batch, batch_idx):
        self.net.train()
        G, spec = batch
        # TODO: push data to GPU
        pred = self.net(G)
        # loss = nn.MSELoss(pred, spec)
        loss = self.loss_fn(pred, spec)
        # TODO: add more metrics as needed
        # TODO: cosine similarity
        # Note that there is redundancy, which is OK
        tensorboard_logs = {'loss': loss}
        tensorboard_logs.update(self._calc_eval_metrics(pred, spec))
        tensorboard_logs = tag(tensorboard_logs, 'train')
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        G, spec = batch
        # TODO: push data to GPU
        pred = self.net(G)
        loss = self.loss_fn(pred, spec)
        # TODO: add more metrics as needed (i.e. AUPR, ...)
        tensorboard_logs = {'loss': loss}
        tensorboard_logs.update(self._calc_eval_metrics(pred, spec))
        tensorboard_logs = tag(tensorboard_logs, 'val')
        return {'valid_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            '--train-library', help='Training library file',
            required=True)
        parser.add_argument(
            '--test-library', help='Testing library file',
            required=True)
        parser.add_argument(
            '--valid-library', help='Validation library file',
            required=True)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--epochs', help='Number of epochs to train.',
            required=False, type=int, default=100)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        # TODO: hash out these args
        parser.add_argument(
            '--hidden-features',
            help='Number of features in the hidden graph layer', type=int,
            default=100,
        )
        return parser
