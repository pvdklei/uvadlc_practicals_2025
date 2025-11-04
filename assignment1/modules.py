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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params: dict[str, np.ndarray | None] = {'weight': None, 'bias': None} # Model parameters
        self.grads: dict[str, np.ndarray | None] = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        if input_layer:
            self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(1. / in_features)
        else:
            self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)

        self.params['bias'] = np.zeros(out_features)

        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros(out_features)

        self.x = None  

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        assert self.params['weight'] is not None
        assert self.params['bias'] is not None

        self.x = x

        out = np.matmul(x, self.params['weight'].T) + self.params['bias']

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        assert self.x is not None, "Cannot backpropagate before calling forward."
        assert self.params['weight'] is not None

        db = np.sum(dout, axis=0)
        dw = np.matmul(dout.T, self.x)
        dx = np.matmul(dout, self.params['weight'])

        self.grads['bias'] = db
        self.grads['weight'] = dw

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = x

        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        assert self.x is not None, "Cannot backpropagate before calling forward."

        dx = np.where(self.x > 0, 1, self.alpha * np.exp(self.x)) * dout

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.exp_x = None
        self.sum_exp_x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        out = exp_x / sum_exp_x

        self.x = x
        self.exp_x = exp_x
        self.sum_exp_x = sum_exp_x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        assert self.exp_x is not None, "Cannot backpropagate before calling forward."
        assert self.sum_exp_x is not None, "Cannot backpropagate before calling forward."

        dxdy = - (self.exp_x[:, :, np.newaxis] * self.exp_x[:, np.newaxis, :]) / (self.sum_exp_x[:, :, np.newaxis] ** 2) 
        diag_dxdy = self.exp_x / self.sum_exp_x 
        dxdy += np.eye(dout.shape[1])[np.newaxis, :, :] * diag_dxdy[:, :, np.newaxis]
        dx = np.sum(dxdy * dout[:, :, np.newaxis], axis=1)

        #######################
        # END OF YOUR CODE    # 
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.exp_x = None
        self.sum_exp_x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        y_one_hot = np.zeros_like(x)
        y_one_hot[np.arange(x.shape[0]), y] = 1

        out = - np.sum(np.log(x) * y_one_hot) / x.shape[0]

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        y_one_hot = np.zeros_like(x)
        y_one_hot[np.arange(x.shape[0]), y] = 1

        dx = - (y_one_hot / x) / x.shape[0]

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx