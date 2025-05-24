#!/bin/bash

echo "Starting self-supervised pretraining..."

python train.py --config configs/cifar10_train_epochs200_bs256.yaml --alpha 1.0 --beta 0.0



echo "Starting linear evaluation..."
python train.py --config configs/cifar_eval.yaml --encoder_ckpt ./logs/exman-train.py/runs/000001/checkpoint-39000.pth.tar

echo "Training and evaluation completed!" 