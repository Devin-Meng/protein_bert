#! /bin/bash

python preprocess.py --mask_path=$1
python main.py --output_path=$2
