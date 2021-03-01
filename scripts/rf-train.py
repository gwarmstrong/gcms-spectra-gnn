from gcms_spectra_gnn.tabular import SklearnTrainer, \
    model_to_morgan_fingerprint, collate_arrays
from gcms_spectra_gnn.trainer import MoleculeJSONDataModule
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import argparse


def main(args):

    model = MultiOutputRegressor(
       RandomForestRegressor(
          random_state=724,
       )
    )

    trainer = SklearnTrainer()

    molecule_dataset = MoleculeJSONDataModule(args)
    molecule_dataset.input_transform = model_to_morgan_fingerprint
    molecule_dataset.collate_fn = collate_arrays

    trainer.fit(model, molecule_dataset)


if __name__ == '__main__':
    # CLA INTERFACE NEEDS DEBUGGING / REFACTORING
    parser = argparse.ArgumentParser(add_help=False)
    # options include ddp_cpu, dp, ddp

    hparams = parser.parse_args()
    main(hparams)
