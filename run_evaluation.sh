# /checkpoint-5-20

#  /mnt/workspace/rankcse_wxt/RankCSE-master/runs/checkpoint-5-20
# /mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/dclr/DCLR/result/my-unsup-simcse-bert-base-uncased

python evaluation.py \
    --model_name_or_path /mnt/nfs-storage-pvc-n28/user_codes/rizeJin/zzk/exp/dclr/DCLR/result/my-unsup-simcse-bert-base-uncased \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test