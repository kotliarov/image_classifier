#!/bin/bash
python train.py adjust-learn-rate \
 --data-path=flowers \
 --checkpoint-path="$1" \
 --gpu \
 --arch-name=densenet161 \
 --hidden-layers 512 \
 --output-size=102 \
 --dropout=0.3 \
 --batch-size=96 \
 --max-learning-rate=0.02 \
 --epochs-per-cycle=4 \
 --num-cycles=6 \
 --momentum=0.8 \
 --dump-koeff=0.9
