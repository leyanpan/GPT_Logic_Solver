# HF_SAT
SAT Solving LLMs using the HuggingFace library.

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
python gpt_train.py configs/[YOUR TRAINING CONFIG].py
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


