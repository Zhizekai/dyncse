import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

# from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy
from torch.nn.functional import normalize

# Set path to SentEval
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock
from rankcse.teachers import Teacher

# RL
from rankcse.Agent_4 import PolicyNet, Critic, ReplayMemory, optimize_model
import string

PUNCTUATION = list(string.punctuation)
logger = logging.get_logger(__name__)
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp  # temperature 温度
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    散度
    """

    def __init__(self, beta_):
        super(Divergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """

    def __init__(self, tau, gamma_):
        super(ListNet, self).__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float("-inf")), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float("-inf")), dim=-1)
        loss = -(q * p).nansum() / q.nansum()
        return self.gamma_ * loss

class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the liklihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher)
    """

    def __init__(self, tau, gamma_):
        super(ListMLE, self).__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_
        self.eps = 1e-7

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        y_pred = student_top1_sim_pred
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))


class CLTrainer(Trainer):
    # def __init__(self, model,args,train_dataset,tokenizer,data_collator,bert_data_collator=None, bert_tokenizer=None):
    #     super().__init__(
    #         model=model,
    #         args = args,train_dataset = train_dataset,
    #         tokenizer = tokenizer,
    #         data_collator=data_collator)
    #     self.bert_tokenizer = bert_tokenizer


    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [" ".join(s) for s in batch]

            sentences = [s + " ." if s.strip()[-1] not in PUNCTUATION else s for s in sentences]
            # bert
            sentences = ["""This sentence : " """ + s + """ " means [MASK] .""" for s in sentences]
            # roberta
            # sentences = ["""This sentence : " """ + s + """ " means <mask> .""" for s in sentences]

            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,
            )
            for k in batch:
                # batch[k] = batch[k].to(self.args.device)
                batch[k] = batch[k].unsqueeze(dim=1).to(self.args.device)
            with torch.no_grad():
                _, last_hidden_state = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=False)
            return last_hidden_state.cpu()

        # Set params for SentEval (fastmode)
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ["STSBenchmark", "SICKRelatedness"]
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ["STSBenchmark", "SICKRelatedness", "MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results["STSBenchmark"]["dev"]["spearman"][0]
        sickr_spearman = results["SICKRelatedness"]["dev"]["spearman"][0]
        acc = (stsb_spearman + sickr_spearman) / 2
        rewards = acc - self.model.best_acc
        self.model.first_rewards += 10000 * rewards
        self.model.second_rewards += 10000 * rewards
        if self.model.best_acc:
            self.model.best_acc = (acc + self.model.best_acc) / 2
        else:
            self.model.best_acc = acc
        # rewards = acc
        # self.model.first_rewards += 10000 * rewards
        # self.model.second_rewards += 10000 * rewards

        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}
        logger.info(metrics)
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                avg_transfer += results[task]["devacc"]
                metrics["eval_{}".format(task)] = results[task]["devacc"]
            avg_transfer /= 7
            metrics["eval_avg_transfer"] = avg_transfer

        self.log(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        # assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if self.state.best_metric is None or self.state.best_model_checkpoint is None or operator(metric_value, self.state.best_metric):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                # Only save model when it is the best one
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler

                if self.sharded_ddp:
                    self.optimizer.consolidate_state_dict()

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    # deepspeed.save_checkpoint above saves model/optim/sched
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

                # Save the Trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune

                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                self.store_flos()

            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            # Save optimizer and scheduler
            if self.sharded_ddp:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                # deepspeed.save_checkpoint above saves model/optim/sched
                # torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                # with warnings.catch_warnings(record=True) as caught_warnings:
                #     torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                # reissue_pt_warnings(caught_warnings)
                pass

            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # Maybe delete some older checkpoints.
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.

        The main difference between ours and Huggingface's original implementation is that we
        also load model_args when reloading best checkpoints for evaluation.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # print(next(iter(train_dataloader)))
        # input()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(self.args.max_steps % num_update_steps_per_epoch > 0)
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        self.sharded_ddp = False
        if self.sharded_ddp:
            model = ShardedDDP(model, self.optimizer)
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(not getattr(model.config, "gradient_checkpointing", False) if isinstance(model, PreTrainedModel) else True),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        
        # 获取训练数据迭代器对象
        num_examples = self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * self.args.max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            # global_step 是已经完成的update step 的数量
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} " "batches in the first epoch.")

        # 初始化教师模型 RankCSE - Initialize the teacher
        teacher = None
        if self.args.second_teacher_name_or_path is None:
            if "rank" in self.args.first_teacher_name_or_path:
                teacher = AutoModel.from_pretrained(self.args.first_teacher_name_or_path)
                teacher = teacher.to(self.args.device)
            else:
                teacher_pooler = "cls_before_pooler" if ("simcse" in self.args.first_teacher_name_or_path or "diffcse" in self.args.first_teacher_name_or_path) else "avg"
                teacher = Teacher(model_name_or_path=self.args.first_teacher_name_or_path, pooler=teacher_pooler)
            # rank
            sentence_vecs = torch.tensor(np.load(self.model_args.corpus_vecs)).to(teacher.device)
            # sentence_vecs = sentence_vecs.half()
            sentence_vecs = normalize(sentence_vecs, p=2.0, dim=1)
        else:
            # 两个教师模型
            if "rank" in self.args.first_teacher_name_or_path:
                first_teacher = AutoModel.from_pretrained(self.args.first_teacher_name_or_path)
                first_teacher = first_teacher.to(self.args.device)
            else:
                first_pooler = "cls_before_pooler" if ("simcse" in self.args.first_teacher_name_or_path or "diffcse" in self.args.first_teacher_name_or_path) else "avg"
                first_teacher = Teacher(model_name_or_path=self.args.first_teacher_name_or_path, pooler=first_pooler)
            second_pooler = "cls_before_pooler" if ("simcse" in self.args.second_teacher_name_or_path or "diffcse" in self.args.second_teacher_name_or_path) else "avg"
            second_teacher = Teacher(model_name_or_path=self.args.second_teacher_name_or_path, pooler=second_pooler)
            sentence_vecs = torch.tensor(np.load(self.model_args.corpus_vecs)).to(first_teacher.device)
            # 单教师情况
            # sentence_vecs = torch.tensor(np.load(self.model_args.second_corpus_vecs)).to(first_teacher.device)
            # sentence_vecs = sentence_vecs.half()
            sentence_vecs = normalize(sentence_vecs, p=2.0, dim=1)
            if self.model_args.second_corpus_vecs is not None:
                sentence_vecs_2 = torch.tensor(np.load(self.model_args.second_corpus_vecs)).to(second_teacher.device)
                # sentence_vecs = sentence_vecs.half()
                sentence_vecs_2 = normalize(sentence_vecs_2, p=2.0, dim=1)

            

        """ 初始化强化学习模型，策略模型和价值模型 """
        tau = 0.1
        exploration_prob = 0.1
        RL_train = True
        if RL_train:
     

            # Reinforcement Learning DDPG
            # bert 的 embedding_length是768 bert large 是 1024
            embedding_length = 768
            policy_model1 = PolicyNet(2, embedding_length, self.args.device, batch_size=self.args.per_device_train_batch_size).to(self.args.device)  # num_teacher是2，embedding_length是768
            policy_model1_target = PolicyNet(2, embedding_length, self.args.device, batch_size=self.args.per_device_train_batch_size).to(self.args.device)
            Critic_model1 = Critic(128, 2, embedding_length, batch_size=self.args.per_device_train_batch_size).to(self.args.device)
            Critic_model1_target = Critic(128, 2, embedding_length, batch_size=self.args.per_device_train_batch_size).to(self.args.device)

            policy_model2 = PolicyNet(2, embedding_length, self.args.device, batch_size=self.args.per_device_train_batch_size).to(self.args.device)
            policy_model2_target = PolicyNet(2, embedding_length, self.args.device, batch_size=self.args.per_device_train_batch_size).to(self.args.device)
            Critic_model2 = Critic(128, 2, embedding_length, batch_size=self.args.per_device_train_batch_size).to(self.args.device)
            Critic_model2_target = Critic(128, 2, embedding_length, batch_size=self.args.per_device_train_batch_size).to(self.args.device)

            # 初始化模型参数
            policy_model2.load_state_dict(policy_model1.state_dict())
            policy_model1_target.load_state_dict(policy_model1.state_dict())
            policy_model2_target.load_state_dict(policy_model1.state_dict())

            Critic_model2.load_state_dict(Critic_model1.state_dict())
            Critic_model1_target.load_state_dict(Critic_model1.state_dict())
            Critic_model2_target.load_state_dict(Critic_model1.state_dict())
            # policy_model = PolicyNet(2, 768, self.args.device, batch_size=self.args.per_device_train_batch_size).to(self.args.device)  # 策略网络

        else:

            print("load model")

        samplecnt = 5
        INITIAL_MEMORY = 10000
        MEMORY_SIZE = 10 * INITIAL_MEMORY
        first_memory = ReplayMemory(MEMORY_SIZE)
        second_memory = ReplayMemory(MEMORY_SIZE)
        TARGET_UPDATE = 100
        step_counter = 0
        # PSEUDO_EPISODE_LENGTH = 125 # 伪章节长度
        PSEUDO_EPISODE_LENGTH = 5  # 伪章节长度, replay buffer 的深度，积累多少步经验进行训练
        first_previous_state = None
        second_previous_state = None
        step_counter = 0
        first_total_reward = 0
        second_total_reward = 0
        global_step = 0
        decay_steps = 5
        decay_rate = 0.95
        learning_rate = 1e-4
        first_memory.clear()
        second_memory.clear()

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        # 跳过已经训练的epoch
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        # 开始循环训练从epochs_trained 到num_train_epochs，train 函数里面就开始训练，这里面知道训练结束，是完全结束，跑完所有epoch
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)  # 分布式采样器设置epoch
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            # last_inputs = None
            # RL
            total_reward = 0.0
            global_step += 1
            # epoch_iterator 就是train_dataloader
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # 给学生模型准备训练数据，使用教师模型生成的句向量计算相似度矩阵，并传入到inputs['teacher_top1_sim_pred']
                # RankCSE - pass the similarity lists obtained by the teacher in inputs['teacher_top1_sim_pred']
                with torch.no_grad():

                    # 强制使用bert 的tokenizer生成的 input_ids。Read batch inputs 
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    token_type_ids = None
                    if "token_type_ids" in inputs:
                        token_type_ids = inputs["token_type_ids"]
                        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))
                        token_type_ids = token_type_ids.to(self.args.device)

                    batch_size = input_ids.size(0)
                    num_sent = input_ids.size(1)

                    # Flatten input for encoding by the teacher - (bsz * num_sent, len)
                    input_ids = input_ids.view((-1, input_ids.size(-1)))
                    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))

                    input_ids = input_ids.to(self.args.device)
                    attention_mask = attention_mask.to(self.args.device)
                    teacher_inputs = copy.deepcopy(inputs)
                    teacher_inputs["input_ids"] = input_ids
                    teacher_inputs["attention_mask"] = attention_mask
                    if "token_type_ids" in inputs:
                        teacher_inputs["token_type_ids"] = token_type_ids

                    # Encode, unflatten 展平, and pass to student
                    if teacher is not None:
                        # teacher 对象有值，说明只有一个教师模型
                        if "rank" in self.args.first_teacher_name_or_path:
                            teacher_vecs = teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
                            teacher_vecs = teacher_vecs[input_ids == self.tokenizer.mask_token_id]
                            teacher_vecs = normalize(teacher_vecs, p=2.0, dim=1)
                            teacher_vecs = teacher_vecs.view((batch_size, num_sent, teacher_vecs.size(-1)))  # (bs, num_sent, hidden)
                            z1, z2 = teacher_vecs[:, 0], teacher_vecs[:, 1]
                        else:
                            embeddings = teacher.encode(teacher_inputs)
                            embeddings = embeddings.view((batch_size, num_sent, -1))
                            z1, z2 = embeddings[:, 0], embeddings[:, 1]
                        z1T = z1.to(torch.float)
                        z2T = z2.to(torch.float)
                        dist1 = torch.mm(z1T, torch.transpose(sentence_vecs, 0, 1))
                        dist2 = torch.mm(z2T, torch.transpose(sentence_vecs, 0, 1))
                        cos = nn.CosineSimilarity(dim=-1)
                        teacher_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / self.args.tau2
                    else:
                        # teacher 对象为空，说明有两个教师模型，两个教师模型分别在 first_teacher 和 second_teacher 中
                        if "rank" in self.args.first_teacher_name_or_path:
                            token_type_ids = None
                            first_teacher_vecs = first_teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
                            first_teacher_vecs = first_teacher_vecs[input_ids == self.tokenizer.mask_token_id]
                            first_teacher_vecs = normalize(first_teacher_vecs, p=2.0, dim=1)
                            first_teacher_vecs = first_teacher_vecs.view((batch_size, num_sent, first_teacher_vecs.size(-1)))  # (bs, num_sent, hidden)
                            first_teacher_z1, first_teacher_z2 = first_teacher_vecs[:, 0], first_teacher_vecs[:, 1]
                        else:
                            embeddings1 = first_teacher.encode(teacher_inputs)  # embeddings1.shape: torch.Size([384, 768])
                            embeddings1 = embeddings1.view((batch_size, num_sent, -1))  # embeddings1.shape: torch.Size([128, 3, 768])
                            first_teacher_z1, first_teacher_z2 = embeddings1[:, 0], embeddings1[:, 1]  # 第一个教师模型的两个句子（正样本对）的embedding  torch.Size([128, 768])
                        embeddings2 = second_teacher.encode(teacher_inputs)  
                        embeddings2 = embeddings2.view((batch_size, num_sent, -1))
                        second_teacher_z1, second_teacher_z2 = embeddings2[:, 0], embeddings2[:, 1]  # 第二个教师模型的两个句子（正样本对）的embedding

                        # if self.args.fp16:
                        #     first_teacher_z1 = first_teacher_z1.to(torch.float16)
                        #     first_teacher_z2 = first_teacher_z2.to(torch.float16)
                        #     second_teacher_z1 = second_teacher_z1.to(torch.float16)
                        #     second_teacher_z2 = second_teacher_z2.to(torch.float16)

                        z1T = first_teacher_z1
                        z2T = first_teacher_z2
                        dist1 = torch.mm(first_teacher_z1, torch.transpose(sentence_vecs, 0, 1))
                        dist2 = torch.mm(first_teacher_z2, torch.transpose(sentence_vecs, 0, 1))
                        if self.model_args.second_corpus_vecs is not None:
                            second_dist1 = torch.mm(second_teacher_z1, torch.transpose(sentence_vecs_2, 0, 1))
                            second_dist2 = torch.mm(second_teacher_z2, torch.transpose(sentence_vecs_2, 0, 1))
                            # dist1 = (self.args.alpha_ * dist1) + ((1.0 - self.args.alpha_) * second_dist1)
                            # dist2 = (self.args.alpha_ * dist2) + ((1.0 - self.args.alpha_) * second_dist2)
                            inputs["distances3"] = second_dist1
                            inputs["distances4"] = second_dist2

                        cos = nn.CosineSimilarity(dim=-1)
                        # first_teacher_top1_sim 是第一个教师模型正样本对的句子表征的相似度矩阵，first_teacher_top1_sim是第二个教师模型的
                        first_teacher_top1_sim = cos(first_teacher_z1.unsqueeze(1), first_teacher_z2.unsqueeze(0)) / self.args.tau2
                        second_teacher_top1_sim = cos(second_teacher_z1.unsqueeze(1), second_teacher_z2.unsqueeze(0)) / self.args.tau2
                        first_teacher_top1_sim = first_teacher_top1_sim.to(second_teacher_top1_sim.device)
                        
                    inputs["first_teacher_top1_sim_pred"] = first_teacher_top1_sim
                    inputs["second_teacher_top1_sim_pred"] = second_teacher_top1_sim
                    inputs["distances1"] = dist1
                    inputs["distances2"] = dist2
                    inputs["baseE_vecs1"] = z1T
                    inputs["baseE_vecs2"] = z2T
                    inputs["policy_model1"] = policy_model1
                    inputs["policy_model2"] = policy_model2
                    inputs["steps_done"] = step
              
                """ 进行一步训练，计算损失值, 并且梯度回传 """
                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example. model.no_sync() 是多卡训练相关
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    # 已经在training_step() 函数里面进行 loss.backward()了，tr_loss 只是detach() 的loss，
                    # 这个tr_loss 只是作为训练的指标进行记录
                    tr_loss += self.training_step(model, inputs)  
                self._total_flos += self.floating_point_ops(inputs)

                """ 更新学生模型参数，积累到一定梯度才会一起更新参数 """
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps, gradient_accumulation_steps 是梯度累计步，经过这些步之后再更新权重
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):

                    # print("更新参数",step, self.args.gradient_accumulation_steps)
                    # self.args.gradient_accumulation_steps 没有设置，这里默认是1，所以没有梯度累计，每次都会更新参数
                    # 梯度剪枝Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_cuda_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # 更新权重，这些都是trainer 源码里面的东西 Optimizer step
                    # 其实这个train() 方法是原作者从源码里抄了一份，然后再在那个基础上进行修改的，所以看着很大便
                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_cuda_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=[])
                    

                    """ 强化学习部分 智能体和环境交互获得action 和 策略价值  reinforce learning"""
                    with torch.no_grad():
                         # first_states是first_teacher_state，因为上面执行了training_step，training_step里面有compute_loss ，compute_loss里面有对first_states的定义
                        value1_state = model.first_states 
                        value1_state = [s.float().to(self.args.device) for s in value1_state]  # len(value1_state) is 3

                        # 后面还要append 进去action。。。。这个7812 只要batch_size 改了，7812 也要成倍减小
                        # 7812
                        if step < 7812:
                            action_1, weights_1 = policy_model1.take_action(value1_state)
                            next_q_value_1 = Critic_model1(*value1_state, action_1)
                        else:
                            next_q_value_1 = next_q_value_1

                        value2_state = model.second_states
                        value2_state = [s.float().to(self.args.device) for s in value2_state]
                        # 7812
                        if step < 7812:
                            action_2, weights_2 = policy_model2.take_action(value2_state)
                            value2 = Critic_model2(*value2_state, action_2)
                        else:
                            value2 = value2
                        
                    first_rewards = model.first_rewards
                    if first_previous_state is not None:
                        first_next_state = model.first_states
                        # 存入经验回放池
                        first_memory.push(first_previous_state, first_next_state, action_1, weights_1, first_rewards, next_q_value_1)
                    first_previous_state = model.first_states

                    second_rewards = model.second_rewards
                    if second_previous_state is not None:
                        second_next_state = model.second_states
                        second_next_state = [s.float().to(self.args.device) for s in second_next_state]
                        # second_next_state = second_next_state
                        second_memory.push(second_previous_state, second_next_state, action_2, weights_2, second_rewards, value2)
                    second_previous_state = model.second_states  
                    step_counter += 1
                    first_total_reward += first_rewards
                    second_total_reward += second_rewards

                    # step_counter超过PSEUDO_EPISODE_LENGTH，从replay buffer中获取样本，更新模型参数
                    if step_counter >= PSEUDO_EPISODE_LENGTH:
                        """ 更新强化学习的模型参数 """
                        decayed_learning_rate = learning_rate * (decay_rate**global_step)
                        logger.info(
                            f"first_action: {action_1} second_action: {action_2} first_total_reward{first_total_reward} second_total_reward{second_total_reward} first_weights:{weights_1}second_weights:{weights_2} "
                        )
                        # agent4的优化函数 ，这里更新价值网络和策略网络的参数
                        # optimize_model(first_memory, policy_model1,Critic_model1, self.args.device, lr=decayed_learning_rate)
                        optimize_model(first_memory, policy_model1, policy_model1_target, Critic_model1, Critic_model1_target, self.args.device)
                        first_memory.clear()

                        # optimize_model(second_memory, policy_model2,Critic_model2, self.args.device, lr=decayed_learning_rate)
                        optimize_model(second_memory, policy_model2, policy_model2_target, Critic_model2, Critic_model2_target, self.args.device)
                        second_memory.clear()
                        step_counter = 0
                        first_total_reward = 0
                        second_total_reward = 0
                        # RL 7800
                    if step == 7800 and RL_train:
                        torch.save(policy_model1.state_dict(), "./rl_model/policy_model_ddpg_1.pth")
                        torch.save(policy_model2.state_dict(), "./rl_model/policy_model_ddpg_2.pth")
                        torch.save(Critic_model1.state_dict(), "./rl_model/Critic_model_ddpg_1.pth")
                        torch.save(Critic_model2.state_dict(), "./rl_model/Critic_model_ddpg_2.pth")
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=[])
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning("You enabled PyTorch/XLA debug metrics but you don't have a TPU " "configured. Check your training configuration if this is unexpected.")
            if self.control.should_training_stop:
                break

        """ 训练结束，保存模型参数和训练数据 """
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
            if isinstance(self.model, PreTrainedModel):
                # bert
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                # roberta
                # self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args, mask_token_id=tokenizer.mask_token_id)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False)

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        training_step 这一步就是计算loss, 进行梯度回传，但是不更新参数
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()  # 开启train 模式，这个model就是学生模型
        inputs = self._prepare_inputs(inputs)  # 这里面就是把inputs 放到对应的设备上去

        self.use_cuda_amp = False  # amp 是自动混合精度
        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # calling on DS engine (model_wrapped == DDP(Deepspeed(PretrainedModule)))
            self.model_wrapped.module.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # gpu并行计算相关，如果不使用并行计算，encoder 就是 BertForCL 实例对象，也就是学生模型
        encoder = model.module if isinstance(model, torch.nn.DataParallel) else model 

        distances1 = inputs["distances1"]
        distances2 = inputs["distances2"]
        first_teacher_top1_sim = inputs["first_teacher_top1_sim_pred"]
        second_teacher_top1_sim = inputs["second_teacher_top1_sim_pred"]

        # pooler_output torch.Size([128, 3, 768]) last_hidden_state torch.Size([384, 768]) 
        # [128, 3, 768] 当中的3 是 num_sentence，也就是句子数量
        # python 的解构赋值是按照顺序的，按照原始的bert 输出的情况他有可能pooler_output 接收的不是 model 里面的pooler_output，有可能接收的是last_hidden_state
        # 这部分他给改了，具体改的内容可以看models_rl_2 文件， 这个trainer 传进来的model 就是 models_rl_2当中的 BertForCL 或者 RobertaForCL
        pooler_output, _ = model(**inputs)  


        """ 计算对比学习损失函数 Calculate InfoNCE loss """
        temp = self.model_args.temp
        cos = nn.CosineSimilarity(dim=-1)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]  # z1和z2 是原始样本和正样本。 z1 torch.Size([128, 768])
        cos_sim = cos(z1.unsqueeze(1), z2.unsqueeze(0)) / temp  # cos_sim 就是学生模型的句子表征的相似度矩阵 cos_sim: torch.Size([128, 128])
        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss_o = loss_fct(cos_sim, labels)

        
        """ 计算双向边际损失函数, 这个是sncse 里的, 只有有软负样本的时候才用。Calculate BML loss """
        alpha = encoder.alpha
        beta = encoder.beta
        lambda_ = encoder.lambda_
        num_sent = inputs["input_ids"].size(1)
        if num_sent == 3:
            z3 = pooler_output[:, 2]  # 软负样本的词嵌入 Embeddings of soft negative samples 
            temp1 = torch.cosine_similarity(z1, z2, dim=1)  # 正样本对的余弦相似度 Cosine similarity of positive pairs 
            temp2 = torch.cosine_similarity(z1, z3, dim=1)  # 原始样本和软负样本的余弦相似度 Cosine similarity of soft negative pairs
            temp3 = temp2 - temp1  # similarity difference
            loss1 = torch.relu(temp3 + alpha) + torch.relu(-temp3 - beta)  # BML loss
            loss1 = torch.mean(loss1)
            # loss1 = 0  # 消融实验，取消注释就是不使用BML loss
            loss_o += loss1 * lambda_  # BML Loss
       
        
        """ 这一部分都是自定义的损失函数, 后面只在hing loss 当中用到 """
        def _get_ranks(x: torch.Tensor) -> torch.Tensor:
            x_rank = x.argsort(dim=1)
            ranks = torch.zeros_like(x_rank, dtype=torch.float)
            n, d = x_rank.size()

            for i in range(n):
                ranks[i][x_rank[i]] = torch.arange(d, dtype=torch.float).to(ranks.device)
            return ranks

        def cal_spr_corr(x: torch.Tensor, y: torch.Tensor):
            x_rank = _get_ranks(x)
            y_rank = _get_ranks(y)
            x_rank_mean = torch.mean(x_rank, dim=1).unsqueeze(1)
            y_rank_mean = torch.mean(y_rank, dim=1).unsqueeze(1)
            xn = x_rank - x_rank_mean
            yn = y_rank - y_rank_mean
            x_var = torch.sqrt(torch.sum(torch.square(xn), dim=1).unsqueeze(1))
            y_var = torch.sqrt(torch.sum(torch.square(yn), dim=1).unsqueeze(1))
            xn = xn / x_var
            yn = yn / y_var
            return torch.mm(xn, torch.transpose(yn, 0, 1))
        
        cos_sim_baseE = cal_spr_corr(distances1, distances2)  # 计算斯皮尔曼相关性
        if self.model_args.second_corpus_vecs is not None:
            distances3 = inputs["distances3"]
            distances4 = inputs["distances4"]
            cos_second_sim_baseE = cal_spr_corr(distances3, distances4)
            cos_sim_baseE = (self.args.alpha_ * cos_sim_baseE) + ((1.0 - self.args.alpha_) * cos_second_sim_baseE)
        cos_sim_baseE = cos_sim_baseE.to(cos_sim.device)
        loss_fct_baseE = nn.MSELoss(reduction="none")
        cos_sim_baseE_bound = torch.logical_and(cos_sim_baseE <= self.model_args.baseE_sim_thresh_upp, cos_sim_baseE >= self.model_args.baseE_sim_thresh_low).type(torch.float).to(cos_sim.device)
        mse = loss_fct_baseE(cos_sim * self.model_args.temp, cos_sim_baseE)
        loss_baseE = torch.sum(mse * cos_sim_baseE_bound) / (torch.sum(cos_sim_baseE_bound) + 1e-8)
        
        
        """ 
        强化学习部分 
        下面开始用策略模型预测
        """
        def get_environment_state(sim_tensor, inputs, z1, z2, cos_sim, encoder):

            """
            sim_tensor 是两个教师模型生成句子表征的相似度矩阵摞一起, inputs 是个大字典
            z1 z2 是学生模型的正样本对的句子表征
            cos_sim 是学生模型的正样本对的句子表征的相似度矩阵,这玩意没用到
            encoder是学生模型

            本来loss 就不能当环境，要不然loss 又是reward 又是 state
            """
            
            # 获取环境状态
            state = []

            x1 = torch.cat([z1.unsqueeze(0), z1.unsqueeze(0)], dim=0)  # torch.Size([2, 128, 768])
            x2 = sim_tensor  # torch.Size([2, 128, 128])

            # 使用上一个循环的kd_loss作为环境之一，如果是第一个循环那就用全0 代替
            # 要用loss 做环境的一部分真的需要慎重考虑！！！
            # encoder.kd_loss_for_RL = None
            # if encoder.kd_loss_for_RL is None:
            #     concatenated_loss = torch.zeros(1, 2).to(encoder.device)
            # else:
            #     concatenated_loss = torch.cat([encoder.kd_loss_for_RL.unsqueeze(0), encoder.kd_loss_for_RL.unsqueeze(0)], dim=0)
            #     concatenated_loss = concatenated_loss.unsqueeze(0)  # torch.Size([1, 2])
            # x3 = concatenated_loss  

            # 学生模型的句子表征的相似度矩阵 torch.size([2,128, 128])
            x3 = torch.cat([cos_sim.unsqueeze(0), cos_sim.unsqueeze(0)], dim=0)

            state.append(x1)
            state.append(x2)
            state.append(x3)
            return state

        steps_done = inputs["steps_done"]
        with torch.no_grad():
            """
            inputs.keys()是
            dict_keys(['input_ids', 'attention_mask', 'first_teacher_top1_sim_pred', 'second_teacher_top1_sim_pred', 'distances1', 'distances2',
             'baseE_vecs1', 'baseE_vecs2', 'policy_model1', 'policy_model2', 'steps_done', 'sim_tensor1', 'sim_tensor2'])
            """
           
            sim_tensor1 = torch.cat([first_teacher_top1_sim.unsqueeze(0), second_teacher_top1_sim.unsqueeze(0)], dim=0)
            sim_tensor2 = torch.cat([second_teacher_top1_sim.unsqueeze(0), first_teacher_top1_sim.unsqueeze(0)], dim=0)

            first_teacher_state  = get_environment_state(sim_tensor1, inputs, z1, z2, cos_sim, encoder)
            second_teacher_state = get_environment_state(sim_tensor2, inputs, z1, z2, cos_sim, encoder)
            first_teacher_policy  = inputs["policy_model1"]
            second_teacher_policy = inputs["policy_model2"]

            # 利用训练之后的策略网络生成action，这个action是用来融合相似度矩阵
            # 7812  这里很有可能是因为这里他不够了,最后一个step 不能用了，所以直接把后面的数据跳过了
            if steps_done < 7812:
                # avg_probability 是take_action 返回的weights，也就是[alpha,beta]
                first_action, first_avg_probability = first_teacher_policy.take_action(first_teacher_state)
                second_action, second_avg_probability = second_teacher_policy.take_action(second_teacher_state)
                model.first_states  = first_teacher_state
                model.second_states = second_teacher_state
            else:
                # first_action  = model.first_actions
                # second_action = model.second_actions

                first_action  = 0
                second_action = 0
                first_avg_probability  = model.first_weights
                second_avg_probability = model.second_weights

        if first_action == 0 and second_action == 0:
            kd_loss = 0 
            kd_loss_1 = 0
            kd_loss_2 = 0

        else:
            # total_probability = first_avg_probability + second_avg_probability
            # weight1 = first_avg_probability / total_probability
            # weight2 = second_avg_probability / total_probability

            # RL 的损失函数，计算的是两个动作的权重和与1的差值，消融实验中要把这个注释掉
            total_probability = first_action + second_action
            weight1 = first_action / total_probability
            weight2 = second_action / total_probability

            teacher_top1_sim_pred = (weight1 * first_teacher_top1_sim) + (weight2 * second_teacher_top1_sim) # 最后通过权重计算得到教师模型的top1相似度
            student_top1_sim_pred = cos_sim.clone()  # cos_sim 是学生模型生成的句子表征的相似度矩阵
            
            if self.model_args.distillation_loss == "listnet":
                distillation_loss_fct = ListNet(self.model_args.tau2, self.model_args.gamma_)
            elif self.model_args.distillation_loss == "listmle":
                distillation_loss_fct = ListMLE(self.model_args.tau2, self.model_args.gamma_)
            # 排序蒸馏损失值
            kd_loss = distillation_loss_fct(teacher_top1_sim_pred.to(encoder.device), student_top1_sim_pred)  
            # kd_loss = 0 # 消融实验，解注释就是去掉蒸馏学习的损失函数


            kd_loss_1 = distillation_loss_fct(first_teacher_top1_sim.to(encoder.device), student_top1_sim_pred) 
            kd_loss_2 = distillation_loss_fct(second_teacher_top1_sim.to(encoder.device), student_top1_sim_pred) 
            


        """ 到此所有loss 全部计算完成，之后开始梯度回传和更新参数 """
        model.kd_loss_for_RL = kd_loss  # 使用教师模型和学生模型的蒸馏损失当做环境，给强化学习当做环境用的排序蒸馏损失值

        
        # RankCSE 正样本对之间的句子表征排序要一直 - self-distillation loss
        # div = Divergence(beta_=self.model_args.beta_)
        # sim = Similarity(temp=temp)
        # z1_z2_cos = cos_sim.clone()
        # z2_z1_cos = sim(z2.unsqueeze(1), z1.unsqueeze(0))
        # sd_loss = div(z1_z2_cos.softmax(dim=-1).clamp(min=1e-7), z2_z1_cos.softmax(dim=-1).clamp(min=1e-7))
        
        
        loss = None
        if self.model_args.loss_type == "hinge":
            loss = torch.max(loss_o, self.model_args.baseE_lmb * loss_baseE)
        elif self.model_args.loss_type == "weighted_sum":  
            loss = loss_o + self.model_args.t_lmb * kd_loss
            # 智能体的rewards 是loss 的负数，这个rewards 应该是环境的输出，那么环境就应该是整个训练过程
            model.first_rewards = -kd_loss_1 * 0.5  
            model.second_rewards = -kd_loss_2 * 0.5
         
        else:
            raise NotImplementedError
        return loss
