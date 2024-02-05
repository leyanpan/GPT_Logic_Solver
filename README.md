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
To organize training arguments for training pipelines, various config files and located in the `configs\` diretory. They are composed of individual configuration files as python files that include custom training hyperparameters. These typically include the training dataset, model save path (without timestamp), and context size. To create a new training pipeline (e.g. for a new dataset), create a new config file in the `configs\` directory that includes all the custom parameters for the training process.

The detailed list of configurable training hyperparameters can be view in `gpt_train.py` in the `### Parameters ###` section.

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


