#!/bin/bash

set -x

python train.py \
    --name sgd-10_30 \
    --optimizer sgd

python train.py \
    --name adam-10_30 \
    --optimizer adam

python train.py \
    --name adamw-10_30 \
    --optimizer adamw

python train.py \
    --name sgd-clip_grad-10_30 \
    --optimizer sgd \
    --clip-grad

python train.py \
    --name sgd-steplr-10_30 \
    --optimizer sgd \
    --use-lr-scheduler \
    --lr-step-size 10
