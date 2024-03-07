#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
import onnx


# In[2]:


class my_param_layer(nn.Module):
    def __init__(self, B, M, K, N, chunk_row_size, chunk_col_size):
        super(my_param_layer, self).__init__()
        self.B = B
        self.M = M
        self.K = K
        self.N = N
        self.chunk_row_size = chunk_row_size
        self.chunk_col_size = chunk_col_size
    def my_split(self, A, height, width, dh, dc):
        ls = []
        for i in range(0, height, dh):
            tmp = []
            for j in range(0, width, dc):
                tmp.append(A[i:i+dh, j:j+dc])
            ls.append(tmp)
        return ls
    def matmul(self, A):
        subAs = self.my_split(A, self.M, self.K, self.chunk_row_size, self.chunk_col_size)
        subBs = self.my_split(self.B, self.K, self.N, self.chunk_col_size, self.chunk_row_size)

        # res = torch.zeros(self.M, self.N)
        final_result = []
        for i in range(0, self.M//self.chunk_row_size):
            row_result = []
            for j in range(0, self.N//self.chunk_row_size):
                psum = torch.zeros(self.chunk_row_size, self.chunk_row_size)
                for k in range(0, self.K//self.chunk_col_size):
                    psum += torch.matmul(subAs[i][k], subBs[k][j])
                row_result.append(psum)
                # res[i*self.chunk_row_size:(i+1)*self.chunk_row_size, j*self.chunk_row_size:(j+1)*self.chunk_row_size] = psum
            row_result = torch.cat(row_result, dim = 1)
            final_result.append(row_result)
        final_result = torch.cat(final_result, dim = 0)
        return final_result

    def forward(self, A):
        return self.matmul(A)

M, K, N = 128, 128, 128
chunk_row_size, chunk_col_size = 64, 64
input = torch.randn(M, K)
weight = torch.randn(N, K)
model = my_param_layer(weight.t(), M, K, N, chunk_row_size, chunk_col_size)
res = model(input)
print('loss: ',torch.sum(res - torch.matmul(input, weight.t())))


# In[3]:


print('==================================================================================================================================================')
print('= HW 2-3-2 create a subgraph (2) as shown in the above diagram for the subgraph (1)                                                              =')
print('==================================================================================================================================================')

print("output file: ", "blocked_linear.onnx")

torch.onnx.export(model, input, "blocked_linear.onnx", verbose=True)

