#!/bin/bash
#SBATCH --job-name="mT5 test"
###SBATCH --mail-user=
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu
#SBATCH --partition=gpu

# enable this when on gpu partition (and NOT on gpu-invest)
###SBATCH --qos=job_gpu_preempt

# alternatively run multiprocess on 6 gtx1080ti gpus with qos job_gpu_preempt (further reduce batch size): only works with opus-mt model

# Activate correct conda environment
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.8.0
eval "$(conda shell.bash hook)"
conda activate data_aug



export PYTHONPATH=. HF_DATASETS_CACHE=/storage/workspaces/inf_fdn/hpc_nfp77/visu/textgen_cache TRANSFORMERS_CACHE=/storage/workspaces/inf_fdn/hpc_nfp77/visu/textgen_cache/models
python -m scripts.run_exp3 --finetune=True --model=google/mt5-small --train_size=500 --eval_size=100 --test_size=100 --seq_length=512 --batch_size=4 --grad_acc_steps=1 --epochs=1

# IMPORTANT:
# Run with                  sbatch run_gen.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=128G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash
