#!/bin/bash

ulimit -n 2500
python /home/Zhaoxu/anaconda3/envs/Pytorch/lib/python3.8/site-packages/torch/distributed/launch.py \
       --nnode=1 --node_rank=0 --nproc_per_node=4 \
       protein_bert/main.py --local_world_size=4 >output/ep5-sav7/train.out 2>output/ep5-sav7/train.err &
