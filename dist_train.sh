#!/bin/bash

ulimit -n 2500
python /home/Zhaoxu/anaconda3/envs/Pytorch/lib/python3.8/site-packages/torch/distributed/launch.py \
       --nnode=1 --node_rank=0 --nproc_per_node=8 \
       main.py --local_world_size=8 >output/ep5-eta5/train.out 2>output/ep5-eta5/train.err &
