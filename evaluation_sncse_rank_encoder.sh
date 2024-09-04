PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/RankCSE-master
#  /mnt/workspace/rankcse_wxt/RankCSE-master/runs/checkpoint-5-20
# /mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/wyy/RankCSE-master/runs/result5/
SEED=61507
python evaluation_rank.py \
    --model_name_or_path $PROJECT_DIR/runs/checkpoint-08-31_score_81.05 \
    --task_set sts \
    --mode test > eval_result.txt
