#!/bin/bash
#SBATCH --job-name="mT5"
###SBATCH --mail-user=
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu

module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.8.0
eval "$(conda shell.bash hook)"
conda activate court_gen

export PYTHONPATH=. HF_DATASETS_CACHE=/storage/workspaces/inf_fdn/hpc_nfp77/visu/textgen_cache TRANSFORMERS_CACHE=/storage/workspaces/inf_fdn/hpc_nfp77/visu/textgen_cache/models
python -m scripts.run_exp3 --finetune=True --model=google/mt5-small --train_size=-1 --eval_size=-1 --test_size=-1 --input_length=2048 --output_length=1024 --total_batch_size=16 --epochs=20 --gm=24 --origin=True
# model = mgpt or google/mt5-small, google/mt5-base, google/mt5-large, google/mt5-xl, google/mt5-xxl
# size = -1 for full dataset

