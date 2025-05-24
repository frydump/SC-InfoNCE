#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.

batch_size=64
lr=3e-5
temp=0.2


WANDB_MODE=disabled torchrun --nproc_per_node=2 --master_port=29888 train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased-bs${batch_size}-lr${lr}-temp${temp} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate $lr \
    --max_seq_length 32 \
    --evaluation_strategy no \
    --pooler_type cls \
    --overwrite_output_dir \
    --mlp_only_train \
    --temp $temp \
    --do_train \
    --loss_type 1 \
    --sc_alpha 1.0 \
    --sc_beta 0.0 \
    --disable_tqdm True \
    "$@"


python evaluation.py \
    --model_name_or_path ./result/my-unsup-simcse-bert-base-uncased-bs${batch_size}-lr${lr}-temp${temp}/checkpoint-7500 \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test

