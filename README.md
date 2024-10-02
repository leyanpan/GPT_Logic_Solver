# HF_SAT
SAT Solving LLMs using the HuggingFace library.

# GPT Logic Solver

This repository contains code for training and evaluating a GPT-based model on a SAT (Satisfiability) problem dataset. The model is trained to classify SAT and UNSAT problems, leveraging the Hugging Face `transformers` library.

## Environment Setup

To set up the environment, use the `environment.yml` file. This file contains the necessary dependencies, including PyTorch, Hugging Face libraries, and other tools.

Run the following command to create and activate the environment:

```bash
conda env create -f environment.yml
conda activate HF_SAT

## Datasets
Each dataset have it's corresponding folder in `datasets/`. The folders typically only contain a script for downloading the dataset. Before using the each dataset, you would need to first switch to the corresponding dataset directory and execute `prepare.py`. For example, to prepare the SAT_6_10 dataset, execute:
```
cd datasets/SAT_6_10
python prepare.py
```

## Training
### Configs
To organize training arguments for training pipelines, various config files are located in the `configs\` diretory. They are composed of individual python files that include custom training hyperparameters. These typically include the training dataset, model save path (without timestamp), and context size. To create a new training pipeline (e.g. for a new dataset), create a new config file in the `configs\` directory that includes all the custom parameters for the training process.

The detailed list of configurable training hyperparameters can be viewed in `gpt_train.py` in the `### Parameters ###` section. The following configurable parameters are commonly used:
`out_dir`: The basepath to save to model. A timestamp will be added to the end of the path to prevent overwrite. To disable timestamp, set `append_timestamp` (also a parameter) to `False`.

`block_size`: the context size of the LLM to be trained.

`dataset`: the training dataset to use. This is typically a directory in `datasets/`

(This method is adapted from nanoGPT)

### gpt_train.py
To train a model on a dataset, the following basic command would typically suffice:
```
python gpt_train.py --model_name llama-70M --train_file path/to/train.txt

```
If you need to have custom hyperparameter configuration for the particular purpose of training, you can also add additional options using the commandline in the following form:
```
python gpt_train.py configs/[YOUR TRAINING CONFIG].py [--VAR_NAME=VAR_VALUE]
```
for example, to set the number of epochs to 12 using the base config file for configs/train_sat_6_10.py:
```
python gpt_train.py configs/train_sat_6_10.py --epochs=12
```

PS: If you get
```
    File "<string>", line 26, in <module>
AssertionError
```
check if you used space instead of `=` for one of the parameters or have any formatting errors. You can dive in `configurator.py` to see exactly where the error occured according to the line number in the error.

## Evaluation
Currently, the evaluation focusing on the accuracy of the model to predict SAT/UNSAT of CNF formulas. To evaluate a model on a dataset:
```
python eval.py --dataset=[Dataset Path] --model_dir=[Your downloaded Model Folder] --num_samples=[Number of Test Samples to Evaluate]
```
Note that the evaluation of a single sample is much slower than training because the model needs to incrementally generate each token and does not use batches (TODO: use batches using eval). Therefore, `num_samples` is typically set to 100 for initial evaluation.

### Batch Evaluation
You can run batch evaluation on multiple .txt files in a folder using the folder_eval.py script.

Example for For Linux
```
./folder_eval.sh ./model_checkpoints/6_10_random_ ./datasets/SAT_var_eval > 6_10_random.txt

```
Example for windows 
```
python folder_eval.py models/sat-llama ./datasets/Large_500k_SAT_11_15_marginal_large results.txt

```

## Running on Multi-GPUs
```
accelerate config
accelerate launch gpt_train.py configs/train_config_ss.py
```
## For operations on Multi-Node/Multi-GPU

Currently we are running jobs on both our own and GT's super-compute cluster on individual compute-engines. However, we have experimented with training jobs on multi-node/multi-GPU, but this is still experimental, and we have not 
thoroughly assessed the impact of these runs on performance if any. 

To execute a multi-node/multi-GPU job run sbatch multi_node

