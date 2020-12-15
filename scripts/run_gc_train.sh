python gc-train.py --train-library ../notebooks/data/preprocessed_micro/index.json --valid-library ../notebooks/data/preprocessed_micro/index.json --test-library ../notebooks/data/preprocessed_micro/index.json --batch-size 1 --output-directory . --gpus 1
python gc-train.py --train-library ${1} --valid-library ${1} --test-library ${1} --batch-size 1 --output-directory . --gpus 1
