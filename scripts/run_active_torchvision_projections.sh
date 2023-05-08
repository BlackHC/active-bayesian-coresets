#!/bin/bash


num_projections=("10")
dataset=$2

if [ $dataset == "cifar10" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "svhn" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "fashion_mnist" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "repeated_mnist" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
fi

for proj in "${num_projections[@]}"; do
    for seed in {0..2}; do
        python -m experiments.torchvision_active_projections --coreset $1 --dataset $dataset --seed $seed --batch_size 25 --budget 125 --gamma 0.7 --num_projections $proj --init_num_labeled 20 --num_features 32 --freq_summary 50 --weight_decay $weight_decay --weight_decay_theta $weight_decay_theta
    done
done

#  ./scripts/run_active_torchvision_projections.sh FW repeated_mnist