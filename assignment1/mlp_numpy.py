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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs: int, n_hidden: list[int], n_classes: int):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        if n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_inputs and n_classes must be positive")
        if any(h <= 0 for h in n_hidden):
            raise ValueError("All hidden dimensions must be positive")

        self.layers: list[LinearModule | ELUModule | SoftMaxModule] = []

        alpha = 1.0  # PyToch default value for ELU, discussed this with the TA's.

        if n_hidden:
            # Input layer
            self.layers.append(LinearModule(
                in_features=n_inputs,
                out_features=n_hidden[0],
                input_layer=True,
            ))
            self.layers.append(ELUModule(alpha))

            # Hidden layers
            for i in range(len(n_hidden) - 1):
                self.layers.append(LinearModule(
                    in_features=n_hidden[i],
                    out_features=n_hidden[i + 1],
                ))
                self.layers.append(ELUModule(alpha))

            # Output layer
            self.layers.append(LinearModule(
                in_features=n_hidden[-1],
                out_features=n_classes,
            ))
        else:
            # No hidden layers, direct input to output
            self.layers.append(LinearModule(
                in_features=n_inputs,
                out_features=n_classes,
                input_layer=True,
            ))  

        self.layers.append(SoftMaxModule())

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x
        for layer in self.layers:
            out = layer.forward(out)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for layer in reversed(self.layers):
            dout = layer.backward(dout) 

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        for layer in self.layers:
            layer.clear_cache()

        #######################
        # END OF YOUR CODE    #
        #######################
