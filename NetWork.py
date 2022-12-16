import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np

""" This script defines the network.
"""
class MyNetwork(nn.Module):
    def __init__(self,
                 resnet_size,
                 num_classes,
                 first_num_filters,
                 ):

        super(MyNetwork, self).__init__()
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        self.layer_start = nn.Conv2d(3, first_num_filters, 3, padding=1, stride=1, bias=False)

        self.batch_norm_celu_start = batch_norm_celu_layer(
                num_features=self.first_num_filters,
                eps=1e-5,
                momentum=0.997,
            )


        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2 ** i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, strides, self.resnet_size, self.first_num_filters))

        self.output_layer = output_layer(filters * 4, self.num_classes)

    def forward(self, inputs):

        outputs = self.layer_start(inputs)

        outputs = self.batch_norm_celu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs


#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_celu_layer(nn.Module):
    """ Perform batch normalization then celu.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_celu_layer, self).__init__()
        ### YOUR CODE HERE

        self.bn = nn.BatchNorm2d(num_features, eps, momentum)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        bn_celu = torch.nn.functional.celu(self.bn(inputs))
        return bn_celu
        ### YOUR CODE HERE

class se_net(nn.Module):

    def __init__(self, input_channels, squeeze_ratio) -> None:
        super(se_net, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.layer_1 = nn.Linear(in_features=input_channels, out_features=round(input_channels/squeeze_ratio))
        self.layer_2 = nn.Linear(in_features=round(input_channels/squeeze_ratio), out_features=input_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.globalAvgPool(inputs)
        output = output.view(output.size(0), -1)
        output = self.layer_1(output)
        output = torch.nn.functional.celu(output, inplace=True)
        output = self.layer_2(output)
        output = self.sigmoid(output)
        output = output.view(output.size(0), output.size(1), 1, 1)
        output = output * inputs

        return output

class se_net2(nn.Module):

    def __init__(self, input_channels, squeeze_ratio) -> None:
        super(se_net2, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.layer_1 = nn.Linear(in_features=input_channels, out_features=round(input_channels/squeeze_ratio))
        self.layer_12 = nn.Linear(in_features=round(input_channels/squeeze_ratio), out_features=round(input_channels/(2*squeeze_ratio)))
        self.layer_2 = nn.Linear(in_features=round(input_channels/(2*squeeze_ratio)), out_features=round(input_channels/squeeze_ratio))
        self.layer_21 = nn.Linear(in_features=round(input_channels/squeeze_ratio), out_features=input_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.globalAvgPool(inputs)
        output = output.view(output.size(0), -1)
        output = self.layer_1(output)
        output = torch.nn.functional.celu(output, inplace=True)
        output = self.layer_12(output)
        output = torch.nn.functional.celu(output, inplace=True)
        output = self.layer_2(output)
        output = torch.nn.functional.celu(output, inplace=True)
        output = self.layer_21(output)
        output = self.sigmoid(output)
        output = output.view(output.size(0), output.size(1), 1, 1)
        output = output * inputs

        return output


class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.layer_1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.layer_2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.residual_projection = None
        #self.bn = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.997)
        self.se_net = se_net2(filters, 4)
        if projection_shortcut is not None:
            ### YOUR CODE HERE
            self.residual_projection = nn.Conv2d(first_num_filters, filters, 1, stride=strides)
            self.layer_1 = nn.Conv2d(first_num_filters, filters, 3, padding=1, stride=strides)

        ### YOUR CODE HERE
        self.bn_celu = batch_norm_celu_layer(filters)
        self.bn_celu_1 = batch_norm_celu_layer(filters)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        output_layer_1 = self.layer_1(inputs)
        output_layer_1_bn_celu = self.bn_celu(output_layer_1)
        out_layer_2 = self.layer_2(output_layer_1_bn_celu)
        out_layer_2_bn = self.bn_celu_1(out_layer_2)
        out_layer_2_bn = self.se_net(out_layer_2_bn)

        if self.residual_projection is not None:
            res_input = self.residual_projection(inputs)
            res_sum = out_layer_2_bn + res_input
        else:
            res_sum = out_layer_2_bn + inputs
        output = torch.nn.functional.celu(res_sum)

        return output
        ### YOUR CODE HERE



class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
	  strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        elements = []
        for i in range(resnet_size):
                if i == 0:
                    if filters == first_num_filters:
                        elements.append(standard_block(filters, 1, strides, filters))
                    else:
                        elements.append(standard_block(filters, 1, strides, filters//2))
                else:
                    elements.append(standard_block(filters, None, 1, filters))

        self.layer_stack = nn.Sequential(*elements)

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        stack = self.layer_stack(inputs)
        return stack
        ### END CODE HERE


class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        num_classes: A positive integer. Define the number of classes.
    """

    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()

        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool2d(1)
        filters1 = filters
        filters1 = filters//4
        self.fully_conn = nn.Linear(filters1, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:

        inputs_1 = inputs
        output = self.pool(inputs_1)
        flattened = output.view(output.size(0), -1)
        flattened = self.dropout(flattened)
        out = self.fully_conn(flattened)
        return out
 
