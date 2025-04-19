CHECKPOINT_DIR=checkpoint-01-06
# /mnt/workspace/rankcse_wxt/RankCSE-master/runs/checkpoint-5-20

# 模型
# /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/sts_model/unsup-simcse-bert-base-uncased
# /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/dyncse/runs/checkpoint-09-08_roberta_large_84.42
# /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/dyncse/runs/checkpoint-08-31_score_81.09
# ./runs/$CHECKPOINT_DIR

# 保存输出模式，在命令的末尾添加如下命令
# --mode test | tee -a ./runs/$CHECKPOINT_DIR/evaluation_result.txt

# debug模式，在python 命令后面添加如下命令
#python -m debugpy --listen 49359 --wait-for-client evaluation_rank.py

SEED=61507
# :> ./runs/$CHECKPOINT_DIR/evaluation_result.txt  # 清空文件
python evaluation_rank.py \
    --model_name_or_path /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/dyncse/runs/checkpoint-08-31_score_81.09 \
    --task_set sts \
    --mode dev 