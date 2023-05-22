#!/bin/bash

# Experiment Plan:
#
# We run 3 trials each (3 seeds)
#
# FashionMNIST, RepeatedMNIST using LeNet
#
# SVHN, CIFAR-10 using ResNet-18
#
# BALD (using PowerBALD, SoftmaxBALD, SoftrankBALD), Random
#
# ACS using FW
#
# on MNISTs: initial set/budget: 20/320 100/400; acquisition batch sizes: 5, 10, 50, 100
#
# on SVHN: initial set: 1000, 3000; budget 8000; acquisition batch sizes: 200, 500, 1000, 2000
#
# on CIFAR-10: initial set: 1000, 4000; budget: 16000; acquisition batch sizes: 200, 500, 1000, 2000, 4000


# We support one script per job index. We use the first argument to the script to determine which job index we are.

JOB_INDEX=$1
shift

CURRENT_INDEX=0

# A function that checks whether the current job index matches the one we are looking for and executes its arguments if so.
function run_job {
    # if the job index is equal "print", then we just print the command
    if [ $JOB_INDEX == "print" ]; then
        echo "$@"
    elif [ $CURRENT_INDEX == $JOB_INDEX ]; then
        "$@"
    fi
    CURRENT_INDEX=$((CURRENT_INDEX+1))
}

# Helper function to execute all acquisition functions given a dataset, batch size, budget and seed.
function run_acq_job {
    dataset=$1
    batch_size=$2
    budget=$3
    seed=$4
    model=$5
    init_num_labeled=$6
    weight_decay=5e-4
    weight_decay_theta=5e-4
    proj=10

    # if the model is LeNet, then we use the _lenet suffix
    if [ $model == "LeNet" ]; then
        model_suffix="_lenet"
        num_features=84
    else
        model_suffix=""
        num_features=128
    fi

    # TK: check num_features parameter. Can we please use the defaults for this one?
    # Check also freq_summary for default use

    # Random
    run_job python -m experiments.torchvision_active$model_suffix --acq None --coreset Random --dataset $dataset \
      --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled \
      --num_features $num_features \
      --weight_decay $weight_decay

    # BALD (using PowerBALD, SoftmaxBALD, SoftrankBALD)
    for acq in "PowerBALD" "SoftmaxBALD" "SoftrankBALD"; do
      run_job python -m experiments.torchvision_active$model_suffix --acq $acq --coreset Stochastic --dataset $dataset \
        --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled \
        --num_features $num_features \
        --weight_decay $weight_decay
    done

    run_job python -m experiments.torchvision_active$model_suffix --acq "BALD" --coreset Argmax --dataset $dataset \
      --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled \
      --num_features $num_features \
      --weight_decay $weight_decay

    run_job python -m experiments.torchvision_active_projections$model_suffix --acq Proj --coreset FW --dataset $dataset \
        --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled \
        --gamma 0.7 --num_projections $proj  \
        --num_features $num_features --weight_decay $weight_decay --weight_decay_theta $weight_decay_theta
}

# We start with the seed in the outer loop, because we want to run different jobs first before running the same job
# with a different seed.
for seed in "1231212" "2139843534" "9438745"; do
  # FashionMNIST, RepeatedMNIST using LeNet
  # MNISTs: initial set/budget: 20/320 100/400; acquisition batch sizes: 5, 25, 50, 100
  for dataset in "fashion_mnist" "repeated_mnist"; do
    for batch_size in "25" "50" "100"; do
      # Arg info
      # dataset=$1
      # batch_size=$2
      # budget=$3
      # seed=$4
      # model=$5
      # init_num_labeled=$6
      run_acq_job $dataset $batch_size 200 $seed "LeNet" 20
      run_acq_job $dataset $batch_size 200 $seed "LeNet" 100
    done
  done
  # on SVHN: initial set: 1000, 3000; budget 8000; acquisition batch sizes: 250, 500, 1000, 2000
  for dataset in "svhn"; do
    for batch_size in "250" "500" "1000" "2000"; do
      run_acq_job $dataset $batch_size 6000 $seed "ResNet18" 1000
      run_acq_job $dataset $batch_size 4000 $seed "ResNet18" 3000
    done
  done
  # on CIFAR-10: initial set: 1000, 4000; budget: 16000; acquisition batch sizes: 250, 500, 1000, 2000, 4000
  for dataset in "cifar10"; do
    for batch_size in "250" "500" "1000" "2000" "4000"; do
      run_acq_job $dataset $batch_size 16000 $seed "ResNet18" 1000
      run_acq_job $dataset $batch_size 12000 $seed "ResNet18" 5000
    done
  done
done

# If job index is print, print the total number of jobs
if [ $JOB_INDEX == "print" ]; then
    echo "Total number of jobs: $CURRENT_INDEX"
fi
