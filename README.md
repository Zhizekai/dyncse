# DynCSE: Dynamic Contrastive Learning of Sentence Embeddings via Reinforcement Learning from Multiple Teachers

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

## Overview
![DynCSE](RLcse_v2.2-overview.png)

## Setups

[![Python](https://img.shields.io/badge/python-3.8.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-386/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). 

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Download the pretraining dataset
```
cd data
bash download_wiki.sh
```

### Download the downstream dataset
```
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Training
(The same as `run_rankcse.sh`.)
```bash
bash run_train_roberta.sh
```

Our new arguments:
* `--<first/second>_teacher_name_or_path`: the model name of of the teachers for distilling ranking information. In the paper, two teachers are used, but we provide functionality for one or two (just don't set the second path).
* `--distillation_loss`: whether to use the ListMLE or ListNet problem formulation for computing distillation loss. 
* `--alpha_`: in the paper, alpha is used to balance the ground truth similarity scores produced from two teachers, ie. alpha * teacher_1_rankings + (1 - alpha) * teacher_2_rankings. Unused if only one teacher is set.
* `--beta_`: weight/coefficient for the ranking consistency loss term
* `--gamma_`: weight/coefficient for the ranking distillation loss term
* `--tau2`: temperature for softmax computed by teachers. If ListNet is set, tau3 is set to tau2 / 2 by default, as suggested in the paper. 


Arguments from [SimCSE](https://github.com/princeton-nlp/SimCSE):
* `--train_file`: Training file path (`data/wiki1m_for_simcse.txt`). 
* `--model_name_or_path`: Pre-trained checkpoints to start with such as BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`).
* `--temp`: Temperature for the contrastive loss. We always use `0.05`.
* `--pooler_type`: Pooling method.
* `--mlp_only_train`: For unsupervised SimCSE-based models, it works better to train the model with MLP layer but test the model without it.

## Evaluation

You can run the commands below for evaluation after using the repo to train a model:

```bash
python evaluation.py \
    --model_name_or_path <your_output_model_dir> \
    --pooler cls_before_pooler \
    --task_set <sts|transfer|full> \
    --mode test
```

For more detailed information, please check [SimCSE's GitHub repo](https://github.com/princeton-nlp/SimCSE).


## Pretrained models

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/perceptiveshawty)

* RankCSE-ListMLE-BERT-base (reproduced): https://huggingface.co/perceptiveshawty/rankcse-listmle-bert-base-uncased
* RankCSE-ListNet-BERT-base (reproduced): https://huggingface.co/perceptiveshawty/rankcse-listnet-bert-base-uncased

We can load the models using the API provided by [SimCSE](https://github.com/princeton-nlp/SimCSE). 
See [Getting Started](https://github.com/princeton-nlp/SimCSE#getting-started) for more information.

