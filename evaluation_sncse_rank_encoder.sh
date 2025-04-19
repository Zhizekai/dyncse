CHECKPOINT_DIR=checkpoint-11-24
#  /mnt/workspace/rankcse_wxt/RankCSE-master/runs/checkpoint-5-20
# /mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/wyy/RankCSE-master/runs/result5/
SEED=61507
:> ./runs/$CHECKPOINT_DIR/evaluation_result.txt  # 清空文件
python evaluation_rank.py \
    --model_name_or_path ./runs/$CHECKPOINT_DIR \
    --task_set sts \
    --mode test | tee -a ./runs/$CHECKPOINT_DIR/evaluation_result.txt