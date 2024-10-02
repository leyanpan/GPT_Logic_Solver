
# Transformer in SAT Solving

This repository contains code for training and evaluating a GPT-based model on SAT (Satisfiability) problem datasets. The model classifies SAT and UNSAT problems using the Hugging Face transformers library.

---

## Environment Setup

To set up the environment, follow these steps:

1. Use the `environment.yml` file to install the necessary dependencies (PyTorch, Hugging Face libraries, etc.).
2. Run the following commands to create and activate the environment:

   ```bash
   conda env create -f environment.yml
   conda activate HF_SAT
   ```

---

## Datasets

Each dataset has its corresponding folder in the `datasets/` directory. These folders typically contain scripts for downloading and preparing the datasets.

To prepare a dataset, navigate to its directory and run `prepare.py`. For example, to prepare the `SAT_6_10` dataset:

   ```bash
   cd datasets/SAT_6_10
   python prepare.py
   ```

---

## Training

### Configuration

Training configurations are stored as Python files in the `configs/` directory. Each file contains custom training hyperparameters, including:

- **Dataset path**
- **Model save path** (with or without a timestamp)
- **Context size** for the model

To create a new training configuration, add a new Python file in `configs/` and define the necessary parameters. A list of commonly used configurable parameters can be found in the `### Parameters ###` section of `gpt_train.py`, including:

- `out_dir`: The base path to save the model. A timestamp is added by default to avoid overwriting; to disable this, set `append_timestamp=False`.
- `block_size`: The context size for the model.
- `dataset`: The training dataset, usually a directory in `datasets/`.

(This method is adapted from nanoGPT.)

### Running Training

To train a model using a basic configuration, use the following command:

   ```bash
   python gpt_train.py --model_name llama-70M --train_file path/to/train.txt
   ```

To use custom hyperparameters, specify them in the command:

   ```bash
   python gpt_train.py configs/[YOUR_TRAINING_CONFIG].py --epochs=12
   ```

If you encounter an `AssertionError`, check for issues like spaces instead of `=` in the parameter assignments. You can debug using `configurator.py` to locate the error.

---

## Evaluation

The evaluation process focuses on measuring the model's accuracy in predicting SAT/UNSAT for CNF formulas.

To evaluate a model on a dataset, use:

   ```bash
   python eval.py --dataset=[Dataset Path] --model_dir=[Model Directory] --num_samples=[Number of Test Samples]
   ```

*Note*: Evaluation can be slower than training since token generation occurs incrementally and without batching. Typically, `num_samples` is set to 100 for initial evaluation.

### Batch Evaluation

For batch evaluation on multiple `.txt` files, use `folder_eval.py`:

- **Linux**:

   ```bash
   ./folder_eval.sh ./model_checkpoints/6_10_random_ ./datasets/SAT_var_eval > 6_10_random.txt
   ```

- **Windows**:

   ```bash
   python folder_eval.py models/sat-llama ./datasets/Large_500k_SAT_11_15_marginal_large results.txt
   ```
