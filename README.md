

python train.py --data_dir data/ \
--model_dir model/ \
--dataset beijing/ \
--num_train_epoches 10 \
--batch_size 100 \
--train_type pretrain \

python train.py --data_dir data/ \
--model_dir model/ \
--dataset beijing/ \
--num_train_epoches 10 \
--batch_size 100 \
--train_type train \

python infer.py --data_dir data/ \
--model_dir model/ \
--model_name st_attn_td_epoch.1 \
--dataset beijing/ \
--infer_type all \
--heuristic heu \


Evaluation on test dataset outputs the following for our examples:

Precision:

Short:  0.821
Medium: 0.757
Long: 0.684


