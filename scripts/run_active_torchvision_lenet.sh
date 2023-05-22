#!/bin/bash


batch_sizes=("25")
init_num_labeled=20
budget=300
dataset=$3

if [ $dataset == "fashion_mnist" ]; then
  weight_decay=5e-4
elif [ $dataset == "repeated_mnist" ]; then
  weight_decay=5e-4
else
  # Error out
  echo "Dataset not supported for LeNet model."
fi

for batch_size in "${batch_sizes[@]}"; do
  for seed in {0..4}; do
    python -m experiments.torchvision_active_lenet --acq $1 --coreset $2 --dataset $dataset --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled --num_features 32 --freq_summary 50 --weight_decay $weight_decay
  done
done
