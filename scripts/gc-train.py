#!/usr/bin/env python3

import argparse

import torch
from pytorch_lightning import Trainer
from gcms_spectra_gnn.trainer import GCLightning, MoleculeJSONDataModule
from gcms_spectra_gnn.transforms import DEFAULT_ELEMENTS


def main(args):
    print('args', args)
    # if args.load_from_checkpoint is not None:
    #     model = GCLightning.load_from_checkpoint(
    #         args.load_from_checkpoint)
    # else:
    model_args = {"input_features": len(DEFAULT_ELEMENTS),
                  "output_features": 481,
                  "hidden_features": args.hidden_features,
                  }
    model = GCLightning(args, model_args)
    # profiler = AdvancedProfiler()

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        num_nodes=args.nodes,
        # check_val_every_n_epoch=1,
        val_check_interval=1.0,
        # set this to true when debugging
        fast_dev_run=False,
        # auto_scale_batch_size='power',
        # profiler=profiler,
    )

    molecule_dataset = MoleculeJSONDataModule(args)
    print('model', model)
    trainer.fit(model, molecule_dataset)

    # In case this doesn't checkpoint
    torch.save(model.state_dict(),
               args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    # CLA INTERFACE NEEDS DEBUGGING / REFACTORING
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    # options include ddp_cpu, dp, ddp
    parser = GCLightning.add_model_specific_args(parser)

    hparams = parser.parse_args()
    main(hparams)
