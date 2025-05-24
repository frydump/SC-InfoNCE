# SimCLR

A PyTorch implementation of [SimCLR framework](https://github.com/google-research/simclr) for self-supervised learning, supporting pre-training and linear evaluation on CIFAR-10 dataset.


## Training Process

Model training consists of two steps: (1) self-supervised encoder pretraining and (2) linear classifier training on encoder representations. Both steps are done with the `train.py` script.

### Self-supervised Pretraining

The config `cifar10_train_epochs200_bs256.yaml` contains the parameters for CIFAR-10 dataset. The pretraining command is:

```(bash)
python train.py --config configs/cifar_train_epochs200_bs256.yaml
```

### Linear Evaluation

To train a linear classifier on top of the pretrained encoder, run:

```(bash)
python train.py --config configs/cifar_eval.yaml --encoder_ckpt <path-to-encoder>
```

### Logs

Training logs and models will be stored at `./logs/exman-train.py/runs/<experiment-id>/`. You can access all experiments from python using: `exman.Index('./logs/exman-train.py').info()`.
