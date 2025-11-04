################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    preds = torch.argmax(predictions, dim=1)
    correct = torch.sum(preds == targets).item()
    accuracy = correct / targets.size(0)

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model: MLP, data_loader: torch.utils.data.DataLoader) -> float:
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    total_correct = 0
    total_samples = 0
    device = model.device

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.reshape(inputs.shape[0], -1)  # Flatten inputs
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_correct += torch.sum(torch.argmax(outputs, dim=1) == targets).item()
            total_samples += targets.size(0)

    avg_accuracy = total_correct / total_samples

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims: list[int], lr: float, use_batch_norm: bool, batch_size: int, epochs: int, seed: int, data_dir: str) -> tuple[MLP, list[float], float, utils.LoggingDict]:
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # Note: there was a type here. 
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():  # Added this as well for MPS
        torch.mps.manual_seed(seed)

    # Set default device (I changed this to also support MPS)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(
        n_inputs=3 * 32 * 32,
        n_hidden=hidden_dims,
        n_classes=10,
        use_batch_norm=use_batch_norm,
    )
    loss_module = nn.CrossEntropyLoss()
    model.to(device)
    loss_module.to(device)
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_val_accuracy = 0.0
    best_model = None
    val_accuracies = []
    training_losses = []
    for epoch in tqdm(range(epochs), desc="Training epochs", position=0, leave=True):
        model.train()
        epoch_losses = []
        for inputs, targets in tqdm(cifar10_loader["train"], desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False):
            inputs = inputs.reshape(inputs.shape[0], -1).to(device)  # Flatten inputs
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_module(outputs, targets)

            epoch_losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Store epoch average loss instead of all batch losses
        training_losses.append(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_accuracy = evaluate_model(model, cifar10_loader["validation"])
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)        

    # TODO: Test best model
    assert best_model is not None
    test_accuracy = evaluate_model(best_model, cifar10_loader["test"])
    # TODO: Add any information you might want to save for plotting
    logging_dict: utils.LoggingDict = {
        "learning_rate": lr,
        "hidden_dims": hidden_dims,
        "use_batch_norm": use_batch_norm,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "training_losses": training_losses,
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = utils.output_dir(f"mlp_pytorch_{timestamp}")
    utils.save_accuracies_plot(output_dir, val_accuracies, test_accuracy)
    utils.save_loss_plot(output_dir, logging_dict.get("training_losses", []))
    utils.save_results(output_dir, val_accuracies, test_accuracy, logging_dict)
