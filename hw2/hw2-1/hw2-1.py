#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
import torchinfo
import torch
# 加載 GoogLeNet 模型
model = models.googlenet(pretrained=True)
print(model)

input_shape = (3, 224, 224)


# In[2]:


print('==================================================================================================================================================')
print('= HW 2-1-1 Calculate the number of model parameters                                                                                              =')
print('==================================================================================================================================================')
# Calculating the total number of parameters in the model
total_params = sum(param.numel() for param in model.parameters())
print("Total number of parameters: ", total_params)


# In[3]:


print('==================================================================================================================================================')
print('= HW 2-1-2 Calculate memory requirements for storing the model weights                                                                           =')
print('==================================================================================================================================================')
param_size = sum(param.numel() * param.element_size() for param in model.parameters())
print("Total memory requirement: {} MB".format(param_size/(1024*1024)))


# In[4]:


print('==================================================================================================================================================')
print('= HW 2-1-3 Use Torchinfo to print model architecture including the number of parameters and the output activation size of each layer             =')
print('==================================================================================================================================================')
# torchinfo.summary(model, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0)
torchinfo.summary(model, (3, 224, 224), batch_dim=0, col_names=("output_size", "num_params"), verbose=0)


# In[5]:


def calculate_output_shape(input_shape, layer):
    # Calculate the output shape for Conv2d, MaxPool2d, and Linear layers
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        stride = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        padding = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        dilation = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )

        output_height = (
            input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        output_width = (
            input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        return (
            layer.out_channels if hasattr(layer, "out_channels") else input_shape[0],
            output_height,
            output_width,
        )
    elif isinstance(layer, nn.Linear):
        # For Linear layers, the output shape is simply the layer's output features
        return (layer.out_features,)
    else:
        return input_shape


def calculate_macs(layer, input_shape, output_shape):
    # Calculate MACs for Conv2d and Linear layers
    if isinstance(layer, nn.Conv2d):
        kernel_ops = (
            layer.kernel_size[0]
            * layer.kernel_size[1]
            * (layer.in_channels / layer.groups)
        )
        output_elements = output_shape[1] * output_shape[2]
        macs = int(kernel_ops * output_elements * layer.out_channels)
        return macs
    elif isinstance(layer, nn.Linear):
        # For Linear layers, MACs are the product of input features and output features
        macs = int(layer.in_features * layer.out_features)
        return macs
    else:
        return 0


# In[6]:


import torch.nn as nn
input_shape = (3, 224, 224)
total_mac = 0

# Initial input shape
input_shape = (3, 224, 224)
total_macs = 0

# Iterate through the layers of the model
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear)):
        output_shape = calculate_output_shape(input_shape, layer)
        macs = calculate_macs(layer, input_shape, output_shape)
        total_macs += macs
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: {macs}"
            )
        elif isinstance(layer, nn.MaxPool2d):
            # Also print shape transformation for MaxPool2d layers (no MACs calculated)
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: N/A"
            )
        input_shape = output_shape  # Update the input shape for the next layer
print('==================================================================================================================================================')
print('= HW 2-1-4 Calculate computation requirements                                                                                                    =')
print('==================================================================================================================================================')
print(f"Total MACs: {total_macs}")


# In[7]:


import torch

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
# Iterate through the layers of the model
for layer_name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d)):
        layer.register_forward_hook(get_activation(layer_name))
data = torch.randn(1, 3, 224, 224)
output = model(data)

# Access the saved activations
for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer].shape}")

