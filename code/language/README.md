# Unsupervised Sentence Embedding Implementation

This repository contains the implementation of unsupervised sentence embedding learning based on SimCSE's approach. The code is adapted from the paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821).

## Data Preparation

For unsupervised SimCSE, we use 1 million sentences from English Wikipedia. Run the following command to download the data:
```bash
cd data/
bash download_wiki.sh
```

## Training

We provide a single-GPU (or CPU) training script `run_unsup_example.sh`. Key training parameters:

* `--train_file`: Path to training file
* `--model_name_or_path`: Path to pre-trained model (supports BERT and RoBERTa series)
* `--temp`: Temperature for contrastive loss
* `--pooler_type`: Pooling method
* `--mlp_only_train`: Use MLP layer during training but not during testing (recommended for unsupervised SimCSE)
* `--do_mlm`: Whether to use MLM auxiliary objective
* `--mlm_weight`: Weight for MLM objective
* `--mlm_probability`: Masking rate for MLM


### Training Command Example

```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy no \
    --pooler_type cls \
    --overwrite_output_dir \
    --mlp_only_train \
    --temp 0.05 \
    --do_train
```

## Evaluation

1. First, download the evaluation datasets:
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

2. Evaluation command:
```bash
python evaluation.py \
    --model_name_or_path result/my-unsup-simcse-bert-base-uncased \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
```

Evaluation parameters:
* `--model_name_or_path`: Path to model
* `--pooler`: Pooling method
  * `cls_before_pooler`: Use [CLS] token representation without extra linear layer (recommended for unsupervised SimCSE)
  * `cls`: Use [CLS] token representation with linear layer
  * `avg`: Average embeddings of last layer
  * `avg_top2`: Average embeddings of last two layers
  * `avg_first_last`: Average embeddings of first and last layers
* `--mode`: Evaluation mode
  * `test`: Test mode
  * `dev`: Development set evaluation
  * `fasttest`: Fast test mode
