# Understanding InfoNCE: Transition Probability Matrix Induced Feature Clustering

> This repository contains the official implementation of our paper “Understanding InfoNCE: Transition Probability Matrix–Induced Feature Clustering”.

## Overview

This codebase provides implementations for the experiments reported in our paper. It consists of four main experimental modules:

- **Vision**: Visual tasks experiments
- **Language**: Language tasks experiments
- **Graph**: Graph tasks experiments
- **Synthetic**: Synthetic data experiments

Each module includes complete training and evaluation scripts.



## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate InfoNCE
```

(Optional) To enable mixed-precision training with NVIDIA Apex:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 54b93919aadc117cbab1fe5a2af4664bb9842928
pip install -v --disable-pip-version-check --no-cache-dir \
  --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Running Experiments

### Synthetic (Toy)

```bash
cd ./code/synthetic
python main.py
```

###  Training and Evaluation

Navigate to the relevant directory and run the corresponding script.

| Group           | Directory                            | Example Command                |
|-----------------|---------------------------------------|--------------------------------|
| Vision          | `./code/vision/`           | `bash train_eval.sh`   |
| Language        | `./code/language/`        | `bash run_unsup_example.sh`          |
| Graph            | `./code/graph/`           | `bash contrastive.sh`          |

You can reproduce our results by modifying the configuration files in each script.

---

## License

This repository is released solely for anonymous peer review.  
A license will be added upon acceptance.

---

