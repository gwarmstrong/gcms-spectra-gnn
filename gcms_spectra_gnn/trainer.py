import torch
import pytorch_lightning as pl


class GCLightning(pl.LightningModule):

    def __init__(self, args):
        sefl.hparams = args
        pass

    def forward(self, smiles):
        """
        Parameters
        ----------
        smiles : list of str
           Smiles strings
        """
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parent_parser)
        # TODO: hash out these args
        return parser
