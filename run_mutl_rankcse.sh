#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers \
HF_DATASETS_CACHE=$SCRATCH/RankCSE/data/ \
HF_HOME=$SCRATCH/.cache/huggingface \
XDG_CACHE_HOME=$SCRATCH/.cache \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python train.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --t_lmb 0.01 \
    --simf Spearmanr \
    --loss_type weighted_sum \
    --corpus_vecs /home/rizejin/LLM/wxt/RankCSE-master/rankcse/index_vecs_rank/corpus_0.01_sncse.npy \
    --model_name_or_path /home/rizejin/LLM/wxt/RankCSE-master/bert/rank/ \
    --train_file data/corpus.txt \
    --output_dir runs/scratch-listmle-bert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --first_teacher_name_or_path /home/rizejin/LLM/wxt/RankCSE-master/runs/scratch-listmle-bert-base-uncased-11-23 \
    --distillation_loss listmle \
    --alpha_ 0.33 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05