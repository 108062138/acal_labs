#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
import onnx


# In[2]:


class my_layer(nn.Module):
    def __init__(self, B):
        super(my_layer, self).__init__()
        self.B = B
    def matmul(self, A):
        return torch.matmul(A, self.B)
    def forward(self, A):
        return self.matmul(A)

M, K, N = 1, 9216, 4096
chunk_row_size, chunk_col_size = 1, 64
input = torch.randn(M, K)
weight = torch.randn(N, K)

layer = my_layer(weight.t())
res = layer(input)
# correctness
print(torch.sum(res - torch.matmul(input, weight.t())))


# In[6]:


print('==================================================================================================================================================')
print('= HW 2-3-1 create a subgraph (1) that consist of a single Linear layer of size MxKxN                                                             =')
print('==================================================================================================================================================')

print('output file:', "noraml_linear.onnx")

torch.onnx.export(layer, input, "noraml_linear.onnx", verbose=True)

