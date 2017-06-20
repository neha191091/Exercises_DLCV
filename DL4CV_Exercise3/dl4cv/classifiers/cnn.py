import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0, gpu_device = 0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Stride for the convolution layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """

        #conv - relu - 2x2 max pool - fc - dropout - relu - fc

        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "same" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################

        padding = ((height-1)*stride + kernel_size - height)/2

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding).cuda(gpu_device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool).cuda(gpu_device)
        )
        lin_input = num_filters * (height/pool) *(width/pool)
        self.fc_encoder = nn.Sequential(
            nn.Linear(in_features=lin_input, out_features=hidden_dim).cuda(gpu_device),
            nn.Dropout(p=dropout).cuda(gpu_device),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(in_features=hidden_dim, out_features=num_classes).cuda(gpu_device)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        out_conv_encoder = self.conv_encoder(x)
        out_conv_encoder = out_conv_encoder.view(out_conv_encoder.size(0), -1)
        out_fc_encoder = self.fc_encoder(out_conv_encoder)
        out = self.final_fc(out_fc_encoder)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
