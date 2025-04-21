#!/bin/bash
# PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/RankCSE-master
PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/
MODEL_DIR=/mnt/workspace/rankcseKel/RankCSE
SEED=61507
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --model_name_or_path $MODEL_DIR/sts_model/bert-base-uncased/ \
    --train_file $MODEL_DIR/sts_model/corpus/wiki1m_for_simcse.txt \
    --output_dir runs/my-rankcse \
    --num_train_epochs 2 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model stsb_spearman \
    --eval_step 125 \
    --save_steps 250\
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    --first_teacher_name_or_path $MODEL_DIR/sts_model/rank-encoder-sncse-bert-base-uncased/ \
    --second_teacher_name_or_path $MODEL_DIR/sts_model/unsup-simcse-bert-large-uncased/ \
    --distillation_loss listmle \
    --alpha_ 0.67 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
