CHECKPOINT_DIR=checkpoint-01-06
# /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/sts_model/unsup-simcse-bert-base-uncased
# /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/dyncse/runs/checkpoint-09-08_roberta_large_84.42
SEED=61507
# :> ./runs/$CHECKPOINT_DIR/evaluation_result.txt  # 清空文件
#  -m debugpy --listen 49359 --wait-for-client
python  align_uniform.py \
    --model_name_or_path /mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/sts_model/unsup-simcse-bert-base-uncased