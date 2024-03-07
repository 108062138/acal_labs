#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torchvision.models as models
import torch.nn as nn

print('==================================================================================================================================================')
print('= HW 2-3-3 replace the Linear layers in the AlexNet with the equivalent subgraphs                                                                =')
print('==================================================================================================================================================')

batch_size = 64
input_channel = 3
input_height = 224
input_width = 224
input_tensor = torch.randn(batch_size, input_channel, input_height, input_width) # Example input, replace with actual data


# In[7]:


class my_param_layer(nn.Module):
    def __init__(self, B, M, K, N, chunk_row_size, chunk_col_size, last, bias=None):
        super(my_param_layer, self).__init__()
        self.B = B
        self.M = (chunk_row_size - M % chunk_row_size) % chunk_row_size + M
        self.K = (chunk_col_size - K % chunk_col_size) % chunk_col_size + K
        self.N = (chunk_row_size - N % chunk_row_size) % chunk_row_size + N
        self.chunk_row_size = chunk_row_size
        self.chunk_col_size = chunk_col_size
        self.bias = bias
        self.last = last
    def my_split(self, A, target_height, target_width, dh, dc):
        # Calculate the current height and width
        current_height, current_width = A.shape

        # Calculate the required padding to reach the target dimensions
        padding_height = max(0, target_height - current_height)
        padding_width = max(0, target_width - current_width)

        # Apply padding to the bottom and right to match the target dimensions
        # Padding format is (left, right, top, bottom)
        padded_A = torch.nn.functional.pad(A, (0, padding_width, 0, padding_height), "constant", 0)

        # Debugging print, can be removed later
        if padded_A.shape != A.shape:
            print(f"Original shape: {A.shape}, padded into: {padded_A.shape}")
            print(f"Target height: {target_height}, Target width: {target_width}")
            print(f"dh: {dh}, dc: {dc}")

        ls = []
        for i in range(0, padded_A.shape[0], dh):
            tmp = []
            for j in range(0, padded_A.shape[1], dc):
                block = padded_A[i:i+dh, j:j+dc]
                tmp.append(block)
            ls.append(tmp)
        return ls
    def matmul(self, A):
        subAs = self.my_split(A,      self.M, self.K, self.chunk_row_size, self.chunk_col_size)
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
            row_result = torch.cat(row_result, dim = 1)
            final_result.append(row_result)
        final_result = torch.cat(final_result, dim = 0)
        return final_result

    def forward(self, A):
        gemm = self.matmul(A)
        if self.bias != None:
            print('inside forward, gemm:',gemm.shape)
            print('inside forward, bias', self.bias.shape)
            if self.bias.shape[0] != gemm.shape[1]:
                gemm[:, :self.bias.shape[0]] += self.bias
                gemm = gemm[:, :self.bias.shape[0]]
            else:
                gemm += self.bias
        return gemm


# In[8]:


alexnet_origin = models.alexnet(pretrained=True)
alexnet_modified = models.alexnet(pretrained=True)

alexnet_modified.classifier[1] = my_param_layer(alexnet_origin.classifier[1].weight.data.t(), batch_size, 9216, 4096, 64, 64, False, alexnet_origin.classifier[1].bias.data)
alexnet_modified.classifier[4] = my_param_layer(alexnet_origin.classifier[4].weight.data.t(), batch_size, 4096, 4096, 64, 64, False, alexnet_origin.classifier[4].bias.data)
alexnet_modified.classifier[6] = my_param_layer(alexnet_origin.classifier[6].weight.data.t(), batch_size, 4096, 1000, 64, 64, True, alexnet_origin.classifier[6].bias.data)


# In[13]:


alexnet_modified.eval()
alexnet_origin.eval()

converted_output = alexnet_modified(input_tensor)
sample_output = alexnet_origin(input_tensor)


# In[12]:


import onnx

print("output file: ", "modified_alexnet.onnx")

torch.onnx.export(alexnet_modified, input_tensor, "modified_alexnet.onnx")


# In[15]:


print('==================================================================================================================================================')
print('= HW 2-3-4 Correctness verification                                                                                                              =')
print('==================================================================================================================================================')

error = torch.sum(torch.abs(converted_output - sample_output))
num_entry = batch_size * input_channel * input_height * input_width

print('sum of each entry abs. diff: {}, which is contributed by {} entries'.format(error, num_entry))

print('converted output: ',converted_output[:2, :2])
print('sample output: ',sample_output[:2, :2])


# In[ ]:




