import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from gcms_spectra_gnn.models import Net
from gcms_spectra_gnn.json_dataset import MoleculeJSONDataset, collate_graphs
from gcms_spectra_gnn.molecule import (
    basic_dgl_transform, OneHotSpectrumEncoder)
from torch.utils.data import DataLoader


class GCLightning(pl.LightningModule):

    def __init__(self, args, model_init_args):
        super(GCLightning, self).__init__()
        """
        Constructs the trainer

        Parameters
        ----------
        args : dict
            Training arguments
        args : dict
            Arguments for initializing the model.
        """
        self.hparams = args
        # TODO: how to init??
        self.net = Net(**model_init_args)
        self.label_transform = OneHotSpectrumEncoder()

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
        train_dataset = MoleculeJSONDataset(
            self.hparams.train_library,
            graph_transform=basic_dgl_transform,
            label_transform=self.label_transform)
        print("train dataset length:", len(train_dataset))
        assert(len(train_dataset) > 0)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1,
            shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_graphs,
        )
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = MoleculeJSONDataset(
            self.hparams.valid_library,
            graph_transform=basic_dgl_transform,
            label_transform=self.label_transform)
        print("valid dataset length:", len(valid_dataset))
        assert(len(valid_dataset) > 0)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=1,
            shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_graphs,
        )
        return valid_dataloader

    def training_step(self, batch, batch_idx):
        self.net.train()
        G, spec = batch
        # TODO: push data to GPU
        pred = self.net(G, G.ndata['mol_ohe'])
        print("-------------------------------------------------")
        print(pred)
        print(spec)
        print("-------------------------------------------------")
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
            self.net.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser)
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
        return parser
