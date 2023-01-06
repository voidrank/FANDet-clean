#!/usr/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE


CONFIG=$1
GPUS=$2
CHECKPOINT=$3
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=10800 \
    test.py $CONFIG --launcher pytorch ${@:3} --eval bbox 