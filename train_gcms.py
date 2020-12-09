﻿import torch
from torch.utils.data import DataLoader
import copy
import pickle
import logging
from datetime import datetime
from math import sqrt

from cormorant.models import CormorantQM9
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():

    # Initialize arguments -- Just
    args = init_argparse('qm9')

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloader
    #args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9', subtract_thermo=args.subtract_thermo,
    #                                                                force_download=args.force_download
    #                                                                )

    #qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    #for dataset in datasets.values():
    #    dataset.convert_units(qm9_to_eV)

    num_species = 33 # the number of atom types
    charge_scale = 92 # the largest atomic charge you see

    # Construct PyTorch dataloaders from datasets
    data_file = '/simons/scratch/jmorton/gcms/gc-ms-databases/data/dataset.pkl'
    dataset = pickle.load(open(data_file, 'rb'))
    train, test, valid = dataset.split()

    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)
    valid_dataset = copy.copy(dataset)

    train_dataset.trim(train)
    test_dataset.trim(test)
    valid_dataset.trim(valid)
    dataloaders = {
        'train': DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn),
        'test': DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           collate_fn=collate_fn),
        'valid': DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)
    }

    # Initialize model
    model = CormorantQM9(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                         args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                         args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                         charge_scale, args.gaussian_mask,
                         args.top, args.input, args.num_mpnn_levels,
                         num_outputs=100, # this needs to change
                         device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()

if __name__ == '__main__':
    main()
