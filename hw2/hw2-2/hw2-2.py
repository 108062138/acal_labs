#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx
# !wget https://github.com/ONNC/onnc-tutorial/raw/master/models/lenet/lenet.onnx 


# In[2]:


import onnx
from numpy import nan
from onnx import shape_inference

onnx_model = onnx.load('mobilenetv2-10.onnx')
inferred_model = shape_inference.infer_shapes(onnx_model)
# collect nodes
nodes = {}
for node in onnx_model.graph.node:
    if node.op_type not in nodes:
        nodes[node.op_type] = [node]
    else:
        nodes[node.op_type].append(node)


# In[3]:


print('==================================================================================================================================================')
print('= HW 2-2-1 model characteristics                                                                                                                 =')
print('==================================================================================================================================================')
for type in nodes:
    print('optype: {} : [{}]'.format(type, len(nodes[type])))


# In[4]:


res = {}
for type in nodes:
    if type == 'Conv' or type == 'Conv2d':
        for node in nodes[type]:
            key = node.name
            # handle height, width, channel
            dims = []
            for dim in node.input:
                try:
                    dims.append(int(dim))
                except:
                    pass
            if len(dims) == 2:
                dims.append(1)
            height, width, channel = dims[0], dims[1], dims[2]

            kernel_shape, dilations, strides, pads, group = nan, nan, nan, nan, nan

            # handle kernel_shape, dilations, strides
            for attribute in node.attribute:
                if attribute.name == "kernel_shape":
                    kernel_shape = attribute.ints
                elif attribute.name == 'dilations':
                    dilations = attribute.ints
                elif attribute.name == 'strides':
                    strides = attribute.ints
                elif attribute.name == 'pads':
                    pads = attribute.ints
                elif attribute.name == 'group':
                    group = attribute.i
            res[key] = {'height': height, 'width':width, 'channel': channel, 'dilation': dilations, 'stride': strides, 'kernel_size': kernel_shape, 'pads': pads, 'group': group}
    elif type == 'Clip':
        for node in nodes[type]:
            key = node.name
            c_min, c_max = nan, nan
            for attribute in node.attribute:
                if attribute.name == 'min':
                    c_min = attribute.f
                elif attribute.name == 'max':
                    c_max = attribute.f
            res[key] = {'min': c_min, 'max': c_max}
    elif type == 'Add':
        for node in nodes[type]:
            key = node.name
            res[key] = {nan}
    elif type == 'GlobalAveragePool':
        for node in nodes[type]:
            key = node.name
            res[key] = {nan}
    elif type == 'Shape':
        for node in nodes[type]:
            key = node.name
            res[key] = {nan}
    elif type == 'Constant':
        for node in nodes[type]:
            key = node.name
            value = nan
            for attribute in node.attribute:
                if attribute.name == 'value':
                    value = attribute.t
            res[key] = {'value': value}
    elif type == 'Gather':
        for node in nodes[type]:
            key = node.name
            axis = nan
            for attribute in node.attribute:
                if attribute.name == 'axis':
                    axis = attribute.i
            res[key] = {'axis': axis}
    elif type == 'Unsqueeze':
        for node in nodes[type]:
            key = node.name
            axes = nan
            for attribute in node.attribute:
                if attribute.name == 'axes':
                    axes = attribute.ints
            res[key] = {'axes': axes}
    elif type == 'Concat':
        for node in nodes[type]:
            key = node.name
            axis = nan
            for attribute in node.attribute:
                if attribute.name == 'axis':
                    axis = attribute.i
            res[key] = {'axis': axis}
    elif type == 'Reshape':
        for node in nodes[type]:
            key = node.name
            res[key] = {nan}
    elif type == 'Gemm':
        for node in nodes[type]:
            key = node.name
            alpha, beta, transB = nan, nan, nan
            for attribute in node.attribute:
                if attribute.name == 'alpha':
                    alpha = attribute.f
                if attribute.name == 'beta':
                    beta = attribute.f
                if attribute.name == 'transB':
                    transB = attribute.i
            res[key] = {'alpha': alpha, 'beta': beta, 'transB': transB}


# In[5]:


def find_shape_by_name(tensor_name, value_infos):
    for vi in value_infos:
        if vi.name == tensor_name:
            shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
            return shape
    return None
conv_layers_info = []
for node in inferred_model.graph.node:
    if node.op_type == "Conv":
        # Initialize dict to store layer info
        layer_info = {'name': node.name, 'input_shape': [], 'output_shape': []}

        # Input shapes
        for input_name in node.input:
            input_shape = find_shape_by_name(input_name, inferred_model.graph.value_info)
            # In case the shape is not found in value_info, it might be an input tensor
            if input_shape is None:
                input_shape = find_shape_by_name(input_name, inferred_model.graph.input)
            layer_info['input_shape'].append(input_shape)

        # Output shapes
        for output_name in node.output:
            output_shape = find_shape_by_name(output_name, inferred_model.graph.value_info)
            layer_info['output_shape'].append(output_shape)

        conv_layers_info.append(layer_info)
        res[node.name]['input_shape'] = layer_info['input_shape']
        res[node.name]['output_shape'] = layer_info['output_shape']
for k in res:
    print('name: ', k, ' =>', res[k])


# In[6]:


from onnx import shape_inference
inferred_model = shape_inference.infer_shapes(onnx_model)

import onnx
from onnx import shape_inference
from os import path
import sys
from tabulate import tabulate
from onnx import onnx_ml_pb2 as xpb2


onnx_model = onnx.load("mobilenetv2-10.onnx", load_external_data=False)
onnx.checker.check_model(onnx_model)

inferred_model = shape_inference.infer_shapes(onnx_model)
print('shape inference complete ...')

def _parse_element(elem: xpb2.ValueInfoProto):
    name = getattr(elem, 'name', "None")
    data_type = "NA"
    shape_str = "NA"
    etype = getattr(elem, 'type', False)
    if etype:
        ttype = getattr(etype, 'tensor_type', False)
        if ttype:
            data_type = getattr(ttype, 'elem_type', 0)
            shape = getattr(elem.type.tensor_type, "shape", False)
            if shape:
                shape_str = "["
                dims = getattr(shape, 'dim', [])
                for dim in dims:
                    vals = getattr(dim, 'dim_value', "?")
                    shape_str += (str(vals) + ",")
                shape_str = shape_str.rstrip(",")
                shape_str += "]"
    return name, data_type, shape_str

def get_valueproto_or_tensorproto_by_name(name: str, graph: xpb2.GraphProto):
    for i, node in enumerate(inferred_model.graph.node):
            if node.name == "":
                inferred_model.graph.node[i].name = str(i)
    input_nlist = [k.name for k in graph.input]
    initializer_nlist = [k.name for k in graph.initializer]
    value_info_nlist = [k.name for k in graph.value_info]
    output_nlist = [k.name for k in graph.output]

    # get tensor data
    if name in input_nlist:
        idx = input_nlist.index(name)
        return graph.input[idx], int(1)
    elif name in value_info_nlist:
        idx = value_info_nlist.index(name)
        return graph.value_info[idx], int(2)
    elif name in initializer_nlist:
        idx = initializer_nlist.index(name)
        return graph.initializer[idx], int(3)
    elif name in output_nlist:
        idx = output_nlist.index(name)
        return graph.output[idx], int(4)
    else:
        print("[ERROR MASSAGE] Can't find the tensor: ", name)
        print('input_nlist:\n', input_nlist)
        print('===================')
        print('value_info_nlist:\n', value_info_nlist)
        print('===================')
        print('initializer_nlist:\n', initializer_nlist)
        print('===================')
        print('output_nlist:\n', output_nlist)
        print('===================')
        return False, 0

def cal_tensor_mem_size(elem_type: str, shape: [int]):
    """ given the element type of the tensor and its shape, and return its memory size.

    Utility.

    Args:
        ttype: the type of the element of the given tensor. format: 'int', ...
        shape: the shape of the given tensor. format: [] of int

    Returns:
        mem_size: int
    """
    # init
    mem_size = int(1)
    # traverse the list to get the number of the elements
    for num in shape:
        mem_size *= num
    # multiple the size of variable with the number of the elements
    # "FLOAT": 1,
    # "UINT8": 2,
    # "INT8": 3,
    # "UINT16": 4,
    # "INT16": 5,
    # "INT32": 6,
    # "INT64": 7,
    # # "STRING" : 8,
    # "BOOL": 9,
    # "FLOAT16": 10,
    # "DOUBLE": 11,
    # "UINT32": 12,
    # "UINT64": 13,
    # "COMPLEX64": 14,
    # "COMPLEX128": 15
    if elem_type == 1:
        mem_size *= 4
    elif elem_type == 2:
        mem_size *= 1
    elif elem_type == 3:
        mem_size *= 1
    elif elem_type == 4:
        mem_size *= 2
    elif elem_type == 5:
        mem_size *= 2
    elif elem_type == 6:
        mem_size *= 4
    elif elem_type == 7:
        mem_size *= 8
    elif elem_type == 9:
        mem_size *= 1
    elif elem_type == 10:
        mem_size *= 2
    elif elem_type == 11:
        mem_size *= 8
    elif elem_type == 12:
        mem_size *= 4
    elif elem_type == 13:
        mem_size *= 8
    elif elem_type == 14:
        mem_size *= 8
    elif elem_type == 15:
        mem_size *= 16
    else:
        print("Undefined data type")

    return mem_size



def get_bandwidth(graph: xpb2.GraphProto):
    try:
        mem_BW_list = []
        total_mem_BW = 0
        unknown_tensor_list = []
        # traverse all the nodes
        for nodeProto in graph.node:
            # init variables
            read_mem_BW_each_layer = 0
            write_mem_BW_each_layer = 0
            total_each_layer = 0
            # traverse all input tensor
            for input_name in nodeProto.input:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    input_name, graph)
                # parse the ValueInfoProto/TensorProto
                if proto:
                    if type_Num == 3:
                        dtype = getattr(proto, 'data_type', False)
                        # get the shape of the tensor
                        shape = getattr(proto, 'dims', [])
                    elif type_Num == 1 or type_Num == 2:
                        name, dtype, shape_str = _parse_element(proto)
                        shape_str = shape_str.strip('[]')
                        shape_str = shape_str.split(',')
                        shape = []
                        for dim in shape_str:
                            if dim != '':
                                shape.append(int(dim))
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            input_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, input_name, nodeProto.op_type))
                # calculate the tensor size in btye
                
                read_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)

            # traverse all output tensor
            for output_name in nodeProto.output:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    output_name, graph)
                # parse the ValueInfoProto
                if proto:
                    if type_Num == 2 or type_Num == 4:
                        # name, dtype, shape = utils._parse_ValueInfoProto(proto)
                        name, dtype, shape_str = _parse_element(proto)
                        shape_str = shape_str.strip('[]')
                        shape_str = shape_str.split(',')
                        shape = []
                        for dim in shape_str:
                            if dim != '':
                                shape.append(int(dim))
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            output_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, output_name, nodeProto.op_type))
                # calculate the tensor size in btye
                write_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)

            # cal total bw
            total_each_layer = read_mem_BW_each_layer + write_mem_BW_each_layer

            # store into tuple
            temp_tuple = (nodeProto.name, read_mem_BW_each_layer,
                        write_mem_BW_each_layer, total_each_layer)
            #append it
            mem_BW_list.append(temp_tuple)
            # accmulate the value
            total_mem_BW += total_each_layer

        # display the mem_bw of eahc layer
        columns = ['layer', 'read_bw', 'write_bw', 'total_bw']
        # resort the list
        mem_BW_list = sorted(mem_BW_list,
                             key=lambda Layer: Layer[1],
                             reverse=True)
        print(tabulate(mem_BW_list, headers=columns))
        print(
            '====================================================================================\n'
        )
        # display it
        print(
            "The memory bandwidth for processor to execute a whole model without on-chip-buffer is: \n",
            total_mem_BW, '(bytes)\n',
            float(total_mem_BW) / float(1000000), '(MB)\n')
        # display the unknown tensor
        columns = ['op_name', 'unfound_tensor', 'op_type']
        print(tabulate(unknown_tensor_list, headers=columns))
        print(
            '====================================================================================\n'
        )
    except Exception as e:
        print("[ERROR MASSAGE] Unable to display: " + str(e))
        return False

    return True

#從這裡開始
print("start")
print('==================================================================================================================================================')
print('= HW 2-2-2 Data bandwidth requirement                                                                                                            =')
print('==================================================================================================================================================')
get_bandwidth(inferred_model.graph)


# In[7]:


total_mem = 0
for key in res:
    if 'Conv' in key:
        #print(key)
        output_dims = res[key]['output_shape'][0]
        psum = 1
        for dim in output_dims:
            if dim != 0:
                psum *= dim
        total_mem += psum
print('==================================================================================================================================================')
print('= HW 2-2-3 activation memory storage requirement                                                                                                 =')
print('==================================================================================================================================================')
print( 'residual memory requeriment: {} (MB)'.format(total_mem * 4 / 1024 / 1024))


# In[ ]:




