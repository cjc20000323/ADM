#!/bin/bash
#SBATCH -J mixagcls                # 作业名为 test
#SBATCH -o mix_ag_statistic_100_new.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur --gres=gpu:A100:2
#SBATCH -w inspur-gpu-13

#python ./code/train.py  --n-labeled 10 \
#--data-path ./data/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#--lrmain 0.000005 --lrlast 0.0005                 # 要运行的程序
#python ./code/normal_train.py --n-labeled 10 --data-path ./data/ag_news_csv/ --batch-size 8 --epochs 100
python ./code/train_noaug_cls_statistic.py --gpu 0,1 --n-labeled 100 \
--data-path ./data/agnews_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
--lambda-u 1 --T 0.3 --alpha 16 --mix-layers-set 7 9 12 \
--lrmain 0.000005 --lrlast 0.0005 --seed 0
