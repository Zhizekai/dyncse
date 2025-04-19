#!/bin/bash
# roberta-base  bert-base-uncased
# 第一个教师模型：DefSentPlus-simcse-roberta-base，unsup-simcse-roberta-large
# rank-encoder-sncse-bert-base-uncased，unsup-simcse-bert-large-uncased
# --first_teacher_name_or_path $MODEL_DIR/sts_model/rank-encoder-sncse-bert-base-uncased/
# --second_teacher_name_or_path $MODEL_DIR/sts_model/unsup-simcse-bert-large-uncased/ \

# PROJECT_DIR=/mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/dyncse
PROJECT_DIR=/data/home/wangzhilan/zzk/dyncse/dyncse

# MODEL_DIR=/mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse
MODEL_DIR=/data/home/wangzhilan/zzk/dyncse
CHECKPOINT_DIR=checkpoint-2025-04-18
SEED=61507
CUDA_VISIBLE_DEVICES=0 
# --t_lmb 0.001
# python -m debugpy --listen 49359 --wait-for-client
python  train_1.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --t_lmb 0.1 \
    --simf Spearmanr \
    --loss_type weighted_sum \
    --corpus_vecs $PROJECT_DIR/rankcse/index_vecs_rank1/corpus_0.01_sncse.npy \
    --second_corpus_vecs $PROJECT_DIR/rankcse/simcse_large_index_vecs/corpus_0.01_sncse.npy \
    --model_name_or_path $MODEL_DIR/sts_model/bert-base-uncased/ \
    --train_file $MODEL_DIR/sts_model/corpus/wiki1m_for_simcse.txt \
    --output_dir runs/$CHECKPOINT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model stsb_spearman \
    --eval_step 25 \
    --save_steps 250\
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    --first_teacher_name_or_path $MODEL_DIR/sts_model/rank-encoder-sncse-bert-base-uncased/  \
    --second_teacher_name_or_path $MODEL_DIR/sts_model/unsup-simcse-bert-large-uncased/ \
    --distillation_loss listmle \
    --alpha_ 0.50 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --soft_negative_file $MODEL_DIR/sts_model/corpus/soft_negative_samples.txt

