#!/bin/bash


num_projections=("10")
dataset=$2

if [ $dataset == "fashion_mnist" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "repeated_mnist" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
else
  # Error out
  echo "Dataset not supported for LeNet model."
fi

for proj in "${num_projections[@]}"; do
    for seed in {0..4}; do
        python -m experiments.torchvision_active_projections_lenet --coreset $1 --dataset $dataset --seed $seed --batch_size 25 --budget 300 --gamma 0.7 --num_projections $proj --init_num_labeled 20 --num_features 32 --freq_summary 50 --weight_decay $weight_decay --weight_decay_theta $weight_decay_theta
    done
done
