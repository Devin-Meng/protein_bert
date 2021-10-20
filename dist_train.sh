#!/bin/bash

ulimit -n 4000
python /home/Zhaoxu/anaconda3/envs/Pytorch/lib/python3.8/site-packages/torch/distributed/launch.py \
       --nnode=1 --node_rank=0 --nproc_per_node=4 \
       protein_bert/main.py --local_world_size=4 --expe_name=ep9-converge >output/ep9-converge/train.out 2>output/ep9-converge/train.err &
