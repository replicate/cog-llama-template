#!/bin/bash

python train.py \
    --train_data 70k_samples_prompt.jsonl \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 2 \
    --warmup_ratio 0.03 \
    --weights /src/weights_13 