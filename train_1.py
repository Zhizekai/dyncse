import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
# sncse
import json
import string
from collections import defaultdict
PUNCTUATION = list(string.punctuation)

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING, 
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from rankcse.models_rl_2 import RobertaForCL, BertForCL
from rankcse.trainers_rl_5 import CLTrainer
# from rankcse.trainers_DDPG import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# cpu_num = 2 # 这里设置成你想运行的CPU个数
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # 强化学习参数
    rl_learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "强化学习的学习率"})


    # spearmanr
    baseE_sim_thresh_upp: Optional[float] = field(default=1.1)
    baseE_sim_thresh_low: Optional[float] = field(default=-1.1)
    simf: Optional[str] = field(default=None)
    loss_type: Optional[str] = field(default=None)
    baseE_lmb: Optional[float] = field(default=0.0)
    corpus_vecs: Optional[str] = field(default=None)
    second_corpus_vecs: Optional[str] = field(default=None)
    t_lmb: Optional[float] = field(default=0.0)

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # RankCSE's arguments
    first_teacher_name_or_path: str = field(
        default="voidism/diffcse-bert-base-uncased-sts",
        metadata={
            "help": "The model checkpoint for weights of the first teacher model. The embeddings of this model are weighted by alpha. This can be any transformers-based model; preferably one trained to yield sentence embeddings."
        },
    )
    second_teacher_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights of the second teacher model. If set to None, just the first teacher is used. The embeddings of this model are weighted by (1 - alpha). This can be any transformers-based model; preferably one trained to yield sentence embeddings."
        }
    )

    third_teacher_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights of the third teacher model. If set to None, just the first teacher is used. The embeddings of this model are weighted by (1 - alpha). This can be any transformers-based model; preferably one trained to yield sentence embeddings."
        }
    )
    pretrain_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights of the second teacher model. If set to None, just the first teacher is used. The embeddings of this model are weighted by (1 - alpha). This can be any transformers-based model; preferably one trained to yield sentence embeddings."
        }
    )
    distillation_loss: str = field(
        default="listnet",
        metadata={
            "help": "Which loss function to use for ranking distillation."
        },
    )
    tau2: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax used in ranking distillation (same as tau_2 in paper). When training with the ListMLE loss, tau3 is set to 0.5 * tau2, following the observations stated in Section 5.3. "
        },
    )
    alpha_: float = field(
        default=float(1/3),
        metadata={
            "help": "Coefficient to compute a weighted average of similarity scores obtained from the teachers."

        }
    )
    beta_: float = field(
        default=1.0,
        metadata={
            "help": "Coefficient used to weight ranking consistency loss"
        }
    )
    gamma_: float = field(
        default=0.10,
        metadata={
            "help": "Coefficient used to weight ranking distillation loss"
        }
    )


    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    # SNCSE args
    soft_negative_file: str = field(
        default=None
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def get_data_collater(data_args, model_args , tokenizer):
    
    @dataclass
    class OurDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        do_mlm: bool = False
        mlm_probability: float = 0.15

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # bert_batch
            if self.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    # OurDataCollatorWithPadding(tokenizer) 调用的是init 方法，不是__call__ 方法。 __call__ 方法只有在类实例当中使用
    # 例如 data_collator() 调用的是 __call__ 方法，因为OurDataCollatorWithPadding(tokenizer) 是dataclass修饰的数据类，所以直接调用就行
    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(
        tokenizer=tokenizer, do_mlm=model_args.do_mlm,mlm_probability=data_args.mlm_probability)

    return data_collator


def get_train_dataset(data_args, model_args, tokenizer, datasets ):

    # Read soft negative sample 读取软负样本数据
    file_path = data_args.soft_negative_file
    negation = dict()
    f = open(file_path)
    for line in f:
        line = json.loads(line)
        negation[line[0]] = line[1]

    # Prepare features 只有
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "

        # 是否使用软负样本，这部分逻辑有问题，而且  Different_Prompt 和  Different_Prompt_Negation控制的是一个东西
        if Different_Prompt:
            s1 = '''This sentence : " '''
            s1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s1))
            ss1 = '''This sentence of " '''
            ss1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ss1))
            if 'roberta' in model_args.model_name_or_path:
                s2 = ''' " means <mask> .'''
                s2 = tokenizer.encode(s2)[1:-1]
            elif 'bert' in model_args.model_name_or_path:
                s2 = ''' " means [MASK] .'''
                s2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s2))
            else:
                raise NotImplementedError
            assert s2.count(tokenizer.mask_token_id) == 1

            max_seq_length = data_args.max_seq_length - len(s1) - len(s2) - 2
            max_seq_length1 = data_args.max_seq_length - len(ss1) - len(s2) - 2
            sent_features = defaultdict(list)

            for idx in range(total):

                sentence = examples[sent0_cname][idx]
                if sentence.strip()[-1] not in PUNCTUATION:
                    sentence = sentence + " ."

                tokens0 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                if len(tokens0) > max_seq_length:
                    tokens0 = tokens0[: max_seq_length]
                tokens0 = [tokenizer.cls_token_id] + s1 + tokens0 + s2 + [tokenizer.sep_token_id]
                assert tokens0.count(tokenizer.mask_token_id) == 1

                tokens1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                if len(tokens1) > max_seq_length1:
                    tokens1 = tokens1[: max_seq_length1]
                tokens1 = [tokenizer.cls_token_id] + ss1 + tokens1 + s2 + [tokenizer.sep_token_id]
                assert tokens1.count(tokenizer.mask_token_id) == 1

                sent_features["input_ids"].append([tokens0, tokens1])
                sent_features['attention_mask'].append([[1] * len(tokens0), [1] * len(tokens1)])
                if 'roberta' not in model_args.model_name_or_path:
                    sent_features['token_type_ids'].append([[0] * len(tokens0), [0] * len(tokens1)])

            return sent_features
        elif Different_Prompt_Negation:
            # 使用prompt bert 的方法，具体参考prompt bert的论文
            s1 = '''This sentence : " '''
            s1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s1))
            ss1 = '''This sentence of " '''
            ss1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ss1))


            if 'roberta' in model_args.model_name_or_path:
                # roberta tokenizer
                s2 = ''' " means <mask> .'''
                s2 = tokenizer.encode(s2)[1:-1]

                # bert tokenizer
                # bert_s1 = '''This sentence : " '''
                # bert_s1 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(bert_s1))
                # bert_ss1 = '''This sentence of " '''
                # bert_ss1 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(bert_ss1))

                # bert_s2 = ''' " means [MASK] .'''
                # bert_s2 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(bert_s2))


            elif 'bert' in model_args.model_name_or_path:
                s2 = ''' " means [MASK] .'''
                s2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s2))
            else:
                raise NotImplementedError
            assert s2.count(tokenizer.mask_token_id) == 1

            max_seq_length = data_args.max_seq_length - len(s1) - len(s2) - 2
            max_seq_length1 = data_args.max_seq_length - len(ss1) - len(s2) - 2

            # if 'roberta' in model_args.model_name_or_path:
            #     bert_max_seq_length = data_args.max_seq_length - len(bert_s1) - len(bert_s2) - 2
            #     bert_max_seq_length1 = data_args.max_seq_length - len(bert_ss1) - len(bert_s2) - 2

            sent_features = defaultdict(list)

            for idx in range(total):

                sentence = examples[sent0_cname][idx]
                negation_sentence = negation.get(sentence, "Not " + sentence)

                tokens0 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                if len(tokens0) > max_seq_length:
                    tokens0 = tokens0[: max_seq_length]
                tokens0 = [tokenizer.cls_token_id] + s1 + tokens0 + s2 + [tokenizer.sep_token_id]
                assert tokens0.count(tokenizer.mask_token_id) == 1

                tokens1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                if len(tokens1) > max_seq_length1:
                    tokens1 = tokens1[: max_seq_length1]
                tokens1 = [tokenizer.cls_token_id] + ss1 + tokens1 + s2 + [tokenizer.sep_token_id]
                assert tokens1.count(tokenizer.mask_token_id) == 1

                tokens2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negation_sentence))
                if len(tokens2) > max_seq_length1:
                    tokens2 = tokens2[: max_seq_length1]
                tokens2 = [tokenizer.cls_token_id] + ss1 + tokens2 + s2 + [tokenizer.sep_token_id]
                assert tokens2.count(tokenizer.mask_token_id) == 1

                # if 'roberta' in model_args.model_name_or_path:
                #     # bert tokenizer
                #     bert_tokens0 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sentence))
                #     if len(bert_tokens0) > bert_max_seq_length:
                #         bert_tokens0 = bert_tokens0[: bert_max_seq_length]
                #     bert_tokens0 = [bert_tokenizer.cls_token_id] + bert_s1 + bert_tokens0 + bert_s2 + [bert_tokenizer.sep_token_id]
                #     # print(bert_tokenizer.mask_token_id)
                #     # input()
                #     assert bert_tokens0.count(bert_tokenizer.mask_token_id) == 1

                #     bert_tokens1 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sentence))
                #     if len(bert_tokens1) > bert_max_seq_length1:
                #         bert_tokens1 = bert_tokens1[: bert_max_seq_length1]
                #     bert_tokens1 = [bert_tokenizer.cls_token_id] + bert_ss1 + bert_tokens1 + bert_s2 + [bert_tokenizer.sep_token_id]
                #     assert bert_tokens1.count(bert_tokenizer.mask_token_id) == 1

                #     bert_tokens2 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(negation_sentence))
                #     if len(bert_tokens2) > bert_max_seq_length1:
                #         bert_tokens2 = bert_tokens2[: bert_max_seq_length1]
                #     bert_tokens2 = [bert_tokenizer.cls_token_id] + bert_ss1 + bert_tokens2 + bert_s2 + [bert_tokenizer.sep_token_id]
                #     assert bert_tokens2.count(bert_tokenizer.mask_token_id) == 1


                sent_features["input_ids"].append([tokens0, tokens1, tokens2])
                sent_features['attention_mask'].append([[1] * len(tokens0), [1] * len(tokens1), [1] * len(tokens2)])
                if 'roberta' in model_args.model_name_or_path:
                    # sent_features["bert_input_ids"].append([bert_tokens0, bert_tokens1, bert_tokens2])
                    # sent_features['bert_attention_mask'].append([[1] * len(bert_tokens0), [1] * len(bert_tokens1), [1] * len(bert_tokens2)])
                    sent_features['token_type_ids'].append([[0] * len(tokens0), [0] * len(tokens1), [0] * len(tokens2)])

            return sent_features
        else:
            raise NotImplementedError

  
    # map 数据集到特征
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    return train_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # 获取 model_args, data_args, training_args 参数，TODO 读取参数部分可以拆分出来
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    

    # Setup logging 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # 在多卡中判断是否是主进程，如果是主进程则设置日志等级为info
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    # 加载预训练模型和tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)  # 单独指定tokenizer位置
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        # 这部分应该修改，bert的tokenizer和roberta的tokenizer不一样，TODO 教师模型和学生模型的tokenizer应该分开加载
        # bert 的 toeknizer bert_tokenizer_name
        # bert_tokenizer_name = "/mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/sts_model/bert-base-uncased"
        # bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    mask_dict = {"mask_token": tokenizer.mask_token}
    tokenizer.add_special_tokens(mask_dict)

    # 初始化学生模型对象
    if model_args.model_name_or_path:
        # Set hyperparameters of BML loss
        alpha = 0.1
        beta = 0.5
        if 'roberta' in model_args.model_name_or_path:
            lambda_ = 5e-4
        else:
            lambda_ = 1e-3
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                mask_token_id=tokenizer.mask_token_id,
                alpha=alpha,
                beta=beta,
                lambda_=lambda_,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                mask_token_id=tokenizer.mask_token_id,
                pretrain_model_name_or_path=model_args.pretrain_model_name_or_path,
                alpha=alpha,
                beta=beta,
                lambda_=lambda_,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = get_train_dataset(data_args=data_args, model_args=model_args, tokenizer=tokenizer, datasets=datasets) if training_args.do_train else None
    # Data collator OurDataCollatorWithPadding实例对象
    data_collator = get_data_collater(data_args=data_args, model_args=model_args, tokenizer=tokenizer)

    training_args.first_teacher_name_or_path = model_args.first_teacher_name_or_path
    training_args.second_teacher_name_or_path = model_args.second_teacher_name_or_path
    training_args.third_teacher_name_or_path = model_args.third_teacher_name_or_path


    training_args.tau2 = model_args.tau2
    training_args.alpha_ = model_args.alpha_

    # 训练器，两个教师模型路径只在这里面用到了
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # Set as True if do not take soft negative sample into consideration
    Different_Prompt = False

    # Set as True if take soft negative sample into consideration
    Different_Prompt_Negation = True
    main()