#!/bin/bash

model_checkpoint="$1"
image_dir="$2"
class_names="$3"

for filepath in "$image_dir"/*; 
do 
	python predict.py --image "$filepath" \
                   	  --checkpoint "$model_checkpoint" \
                          --top-k 5 \
                          --class-names "$class_names" \
                          --gpu
done
