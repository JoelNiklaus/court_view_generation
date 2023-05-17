#export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=CourtViewGeneration


# if model is mgpt, then input length needs to be the same as output length

MODEL=$1
INPUT_LENGTH=$2
OUTPUT_LENGTH=$3

/home/groups/deho/miniconda3/envs/court_gen/bin/python -m scripts.run_exp3 \
    --finetune=True \
    --model=$MODEL \
    --train_size=-1 \
    --eval_size=1000 \
    --test_size=1000 \
    --input_length=$INPUT_LENGTH \
    --output_length=$OUTPUT_LENGTH \
    --total_batch_size=16 \
    --epochs=1 \
    --gm=80
