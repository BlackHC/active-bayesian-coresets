# Bayesian Batch Active Learning as Sparse Subset Approximation
This repository contains the code to reproduce the experiments carried out in
[Bayesian Batch Active Learning as Sparse Subset Approximation](https://arxiv.org/abs/1908.02144).

The code has been authored by: Robert Pinsler and Jonathan Gordon.

Additional changes: Parmida Atighehchian and Andreas Kirsch.

## Additional Experiments for Stochastic Batch Acquisition

Caveat: no \beta ablation for now.

FashionMNIST, RepeatedMNIST using LeNet

SVHN, CIFAR-10 using ResNet-18

BALD (using PowerBALD, SoftmaxBALD, SoftrankBALD), Random

ACS using FW

on MNISTs: budget: 320 (400), acquisition batch sizes: 25, 50, 100
initial set 20, 100

on SVHN: budget 8000, acquisition batch sizes: 250, 500, 1000, 2000
initial set 1000, 3000

on CIFAR-10: budget 16000, acquisition batch sizes: 250, 500, 1000, 2000, 4000
initial set 1000, 4000

From the paper: 100 posterior samples, 10 projection dim.

## Dependencies and Data Requirements

This code requires the following:
* Python  >= 3.5
* torch >= 1.0
* torchvision >= 0.2.2
* numpy
* scipy
* sklearn
* pandas
* matplotlib
* gtimer

To run the regression experiments, please [download](http://archive.ics.uci.edu/ml/datasets.php) the UCI regression datasets and place them into ./data.



## GPU Requirements
* The code supports experiments on either GPU or CPU processors.

## Usage
The experiments provided in this code base include active learning on standard vision based datasets (classification)
and UCI datasets (regression). The following experiments are provided (see section 7 of the
paper):

1. Active learning for regression: run the following command 

     ```./scripts/run_active_regression.sh DATASET ACQ CORESET```  

    where 
    - ```DATASET``` may be one of ```{yacht, boston, energy, power, year}``` (determines the dataset to be used).
    - ```ACQ```  may be one of ```{BALD, Entropy, ACS}``` (determines the acquisition function to be used).
    - ```CORESET``` may be one of ```{Argmax, Random, Best, FW}``` (determines the querying strategy to be used)  

    For example, to run the proposed method on the boston dataset, please run:
    
    ```./scripts/run_active_regression.sh boston ACS FW```

    This will automatically generate an experimental directory with the appropriate name, and place results from 40 seeds in
    the directory. Hyper-parameters for the experiments can all be found in the main body of the paper.
    
2. Active learning for regression (with projections -- should be used for large datasets e.g., year and power):
run the following command 

     ```./scripts/run_active_regression_projections.sh DATASET NUM_PROJECTIONS```  

    where 
    - ```DATASET``` may be one of ```{yacht, boston, power, protein, year}``` (determines the dataset to be used).
    - ```NUM_PROJECTIONS```  is an integer (determines the number of samples used to estimate values).  

    For example, to run the proposed method on the year dataset, please run:
    
    ```./scripts/run_active_regression.sh year 10```

    This will automatically generate an experimental directory with the appropriate name, and place results from 40 seeds in
    the directory. Hyper-parameters for the experiments can all be found in the main body of the paper.
    
3. Active learning for classification (using standard active learning methods): run the following command

   ```./scripts/run_active_torchvision.sh ACQ CORESET DATASET ```   
   
   where 
    - ```ACQ```  may be one of ```{BALD, Entropy}``` (determines the acquisition function to be used).
    - ```CORESET``` may be one of ```{Argmax, Random, Best}``` (determines the querying strategy to be used)
    - ```DATASET``` may be one of ```{cifar10, svhn, fashion_mnist}``` (determines the dataset to be used).
    
   For example, to run greedy BALD on CIFAR10, run the following command:
   
   ```./scripts/run_active_torchvision.sh BALD Argmax cifar10```
   
   This will automaticall generate an experimental directory with an appropriate name, and place results from
   5 runs in the directory.

4. Active learning for classification (using projections as in section 5 of the paper): run the following command

    ```./scripts/run_active_torchvision_projections.sh CORESET DATASET ```    

    where 
    -```CORESET``` may be one of ```{Argmax, Random, Best, FW}``` (determines the querying strategy to be used)
    -```DATASET``` may be one of ```{cifar10, svhn, fashion_mnist}``` (determines the dataset to be used).
    
    For example, to run the proposed method on the CIFAR10 dataset, please run:
    
    ```./scripts/run_active_torchvision_projections.sh FW cifar10```
    
    This will automaticall generate an experimental directory with an appropriate name, and place results from
    5 runs in the directory.
    
## Plotting
Code to generate active learning curves as exhibited in the paper is also provided. To generate appropriate learning curves,
please run the command 

    python3 ./scripts/enjoy_learning_curves.py --load_dir=LOAD_DIR --metric=METRIC --eval_at=EVAL_AT --format=FORMAT
    
where
- ```LOAD_DIR``` is the path to a results directory generated by an experiment
- ```METRIC``` may be one of ```{LL, RMSE, Accuracy}```
- ```EVAL_AT``` may be one of ```{num_evals, wt, num_samples}``` (```wt``` signifies wall time)
- ```FORMAT``` may be one of ```{png, pdf}```

Running this script will automatically generate a figure of the learning curve in the directory  


## Citation
If you use this code, please cite our [paper](https://arxiv.org/abs/1908.02144):
```
@article{pinsler2019bayesian,
  title={Bayesian Batch Active Learning as Sparse Subset Approximation},
  author={Pinsler, Robert and Gordon, Jonathan and Nalisnick, Eric and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:https://arxiv.org/abs/1908.02144},
  year={2019}
}
```


## General Structure of the Repositiory

* acs: This directory contains the core code used in the repository. Inside you will find the following subdirectories and files:
    - Baselines: contains implementations of the following baseline methods used for comparison in the paper:
        - K-center greedy (https://arxiv.org/abs/1708.00489)
        - K-medoids (applies K-medoids clustering to generate a batch)
    - acquisition_functions: implementation of all the functions necessary for constructing acquisition functions (e.g.,
    BALD, max-entropy, coreset norm computations)
    - al_data_set: implementation of a special data handling class that supports active learning in PyTorch (e.g., handles
    indexing of train and pool sets, and moves examples from pool to train set when queried).
    -coresets: core file handling the method proposed in the paper. Implements classes such as Coreset (general use for
    active learning querying), Frank-Wolfe coresets, random / argmax acquisition. Also contains implementations of baseline
    methods mentioned above in our active learning framework,
    -model: implements the model classes used in our experiments, including linear  regression, neural networks,
    and approximate Bayesian inference layers such as variational inference, (local) reparametrization, etc.
    utils: collection of utility and probability functionalities required in the code base.
* experiments: Contains python scripts for running experiments in the paper. Has experimental scripts for:
    - linear_regression: simple training script for linear regression models on the UCI datasets.
    - linear_regression_active: runs an active learning experiment (starting with an initial labeled set, and querying
    the pool set in batch mode until budget is exhausted) with a specified model, acquisition function, dataset, and 
    additional hyper-parameters.
    - linear_regression_active_projections: same, but using projection based methods for batch mode querying. Should be 
    used for larger regression datasets such as year and protein.
    - torchvision_active: runs a classification based active learning experiment on one of the torchvision datasets (e.g,
    Fashion-MNIST, SVHN, CIFAR10). Should be used with baseline methods.
    - torchvision_active_projections: same, but with projection based methods for batch mode querying. Should be used 
    with our methods (Frank-Wolf) optimization.
* resnet: Contains code necessary for loading and training ResNet modules.
* scripts: Contains .sh files used to run experiments (see above for more details).
