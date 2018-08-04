#!/bin/bash
python train.py adjust-learn-rate \
 --data-path=flowers \
 --checkpoint-path=trained-models \
 --gpu \
 --arch-name=vgg19 \
 --hidden-layers 4096 2048 2048 \
 --output-size=102 \
 --dropout=0.3 \
 --batch-size=96 \
 --max-learning-rate=0.02 \
 --epochs-per-cycle=4 \
 --num-cycles=6 \
 --momentum=0.8 \
 --dump-koeff=0.9
