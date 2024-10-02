#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=2 # number of nodes
#SBATCH --ntasks-per-node=1 # number of MP tasks
#SBATCH --gres=gpu:2 # number of GPUs per node
#SBATCH -C H100
#SBATCH --cpus-per-task=8 # number of cores per tasks
#SBATCH --account=gts-wl67
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cesposo3@gatech.edu
#SBATCH --time=15:59:00 # maximum execution time (HH:MM:SS)
######################
### Set environment ###
######################
module load nvhpc
module load cuda
module load anaconda3/2022.05.0.1
conda activate /your/path/to/conda/env/here/
cd /your/path/to/project_code/here/HF_SAT/
export GPUS_PER_NODE=2
######################
# Set WandB API key
export WANDB_API_KEY=YOUR_API_KEY_HERE
######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
echo $((SLURM_NNODES * GPUS_PER_NODE))
export LAUNCHER="accelerate launch \
--num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
--num_machines $SLURM_NNODES \
--rdzv_backend c10d \
--main_process_ip $head_node_ip \
--main_process_port 29500 \
"
export SCRIPT="gpt_train.py"
export SCRIPT_ARGS="--model_name=llama-70M --train_file=SAT_11_15_marginal_large_train.txt"
# This step is necessary because accelerate launch does not handle multiline arguments properly
