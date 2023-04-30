export CUDA_VISIBLE_DEVICES=1
# models: mgpt, google/mt5-small, google/mt5-base, google/mt5-large, google/mt5-xl

python -m scripts.run_exp3 \
    --finetune=True \
    --model=mgpt \
    --train_size=1000 \
    --eval_size=100 \
    --test_size=100 \
    --input_length=2048 \
    --output_length=512 \
    --total_batch_size=16 \
    --epochs=1 \
    --gm=80
