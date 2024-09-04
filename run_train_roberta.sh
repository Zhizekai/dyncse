#!/bin/bash
# PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/RankCSE-master
# --soft_negative_file $PROJECT_DIR/outputs/corpus/soft_negative_samples.txt
PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/RankCSE-master
MODEL_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/
SEED=61507
CUDA_VISIBLE_DEVICES=0 \
python train_1.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --t_lmb 0.001 \
    --simf Spearmanr \
    --loss_type weighted_sum \
    --corpus_vecs $PROJECT_DIR/rankcse/index_vecs_rank1/corpus_0.01_sncse.npy \
    --model_name_or_path /mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wzl/model-files/bert-large-uncased/ \
    --train_file $MODEL_DIR/sts_model/corpus/wiki1m_for_simcse.txt \
    --output_dir runs/checkpoint-09-04 \
    --num_train_epochs 4 \
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
    --first_teacher_name_or_path $MODEL_DIR/sts_model/rank-encoder-sncse-bert-base-uncased/ \
    --second_teacher_name_or_path $MODEL_DIR/sts_model/unsup-simcse-bert-large-uncased/ \
    --distillation_loss listmle \
    --alpha_ 0.50 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --soft_negative_file $MODEL_DIR/sts_model/corpus/soft_negative_samples.txt

python evaluation_rank.py \
    --model_name_or_path $PROJECT_DIR/runs/checkpoint-09-04 \
    --task_set sts \
    --mode test