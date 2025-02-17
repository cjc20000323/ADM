#!/bin/bash
#SBATCH -J udaagcls_in_10                # 作业名为 test
#SBATCH -o uda_ag_cls_10_2.out# 屏幕上的输出文件重定向到 test.out
#SBATCH -e uda_ag_cls_10_2.err
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04

#python ./code/train.py  --n-labeled 10 \
#--data-path ./data/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#--lrmain 0.000005 --lrlast 0.0005                 # 要运行的程序
#python ./code/normal_train.py --n-labeled 10 --data-path ./data/ag_news_csv/ --batch-size 8 --epochs 100
python ./code/train_UDA.py --gpu 1 --n-labeled 10 --un-labeled 5000 \
--data-path ./data/agnews_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
--lambda-u 1 --T 0.3 --alpha 16 --mix-layers-set 7 9 12 \
--lrmain 0.000005 --lrlast 0.0005 --seed 0 
