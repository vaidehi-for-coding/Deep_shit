from torch import nn
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Linear, AdaptiveAvgPool2d
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.flatten import Flatten


class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        # print("Inside init ResBlock...")
        super().__init__()
        # use torch.nn.ModuleList and add Conv2D, BatchNorm2d....
        # each ResBlock consists of a sequence of (Conv2D, BatchNorm, ReLU) that is repeated twice.
        self.layers = torch.nn.ModuleList([
            # all layers have a filter size of 3
            Conv2d(in_channels, out_channels, 3, stride, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU(),
            # For the second convolution, no stride is used.
            Conv2d(out_channels, out_channels, 3, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU()
        ])
        # call this in forward method of ResBlock
        self.initial_conv = Conv2d(in_channels, out_channels, 3, stride, padding=(1, 1))

    def forward(self, input_tensor):
        # print("Inside forward ResBlock...")
        # apply a 1x1 convolution to the input with stride and channels set accordingly.
        initial_conv_output = self.initial_conv(input_tensor)
        # set first layer output to input tensor, then run a loop for all layers
        layer_output = input_tensor
        for layer in self.layers:
            layer_output = layer(layer_output)

        forward_output = layer_output + initial_conv_output
        return forward_output


class ResNet(nn.Module):

    def __init__(self):
        print("Inside ResNet...")
        super().__init__()
        # now do the same thing that you did in ResBlock, just change the layer list to what is given Tab 1
        self.layers = nn.ModuleList([
            Conv2d(3, 64, 7, stride=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, stride=2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            AvgPool2d(kernel_size=10),
            Flatten(),
            Linear(in_features=512, out_features=2),
            Sigmoid()
            ])

    def forward(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor
