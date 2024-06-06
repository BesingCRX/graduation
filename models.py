# -*- coding: utf-8 -*-
# Torch
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.autograd import grad
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
# from sklearn.externals
import joblib
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm

from visdom import Visdom

from datasets import HyperX
from utils import grouper, sliding_window, count_sliding_window, \
    camel_to_snake

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    # weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 400)
        kwargs.setdefault('batch_size', 100)
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        epoch = kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
    elif name == 'he':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('learning_rate', 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        epoch = kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('lr', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))
    elif name == 'liu':
        kwargs['supervision'] = 'semi'
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault('epoch', 40)
        lr = kwargs.setdefault('lr', 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault('patch_size', 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'boulch':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('lr', 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']), lambda rec, data: F.mse_loss(rec, data.squeeze()))
    elif name == 'mou':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'myModel_Chen':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = MyModel_Chen(n_bands, n_classes, kwargs['dataset'])
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler',
                      optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=10,
                                                           min_lr=1e-3,
                                                           verbose=True))
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', True)
    kwargs.setdefault('radiation_augmentation', True)
    kwargs.setdefault('mixture_augmentation', True)

    kwargs['center_pixel'] = center_pixel

    # Chen_Model
    kwargs['flip_augmentation'] = True
    kwargs['radiation_augmentation'] = True
    kwargs['mixture_augmentation'] = True
    kwargs['sampling_mode'] = 'ten_fold_cross_validation'
    kwargs['class_balancing'] = True
    kwargs['with_exploration'] = True

    return model, optimizer, criterion, kwargs


class Conv3DTo2D(nn.Module):
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        return x.view(batch_size, channels, depth * height, width)

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Project inputs to query, key, and value
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Reshape proj_value to (batch_size, channels, height * width)
        proj_value = proj_value.view(batch_size, channels, -1)

        # Reshape proj_query and proj_key to (batch_size, channels // 8, height * width)
        proj_query = proj_query.view(batch_size, channels // 8, -1)
        proj_key = proj_key.view(batch_size, channels // 8, -1)

        # Compute attention weights
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = self.softmax(energy)

        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Reshape output
        out = out.view(batch_size, channels, height, width)

        # Combine with original input
        out = self.gamma * out + x

        # 计算注意力热图
        attention_map = attention[0].detach().cpu().numpy()  # 转换为 NumPy 数组

        return out, attention_map

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, kernel_size=1):
        super(SelfAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        out = self.gamma * out + x

        # 计算注意力热图
        attention_map = attention[0].detach().cpu().numpy()  # 转换为 NumPy 数组

        # 获取原始热图的尺寸
        original_height, original_width = attention_map.shape

        # 计算裁剪后的新尺寸
        new_height = min(original_height, 100)
        new_width = min(original_width, 100)

        # 裁剪热图
        cropped_attention_map = attention_map[:new_height, :new_width]

        # 返回注意力热图和其他数据
        return out, cropped_attention_map


class MyModel_Chen(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, output_units, dataset):
        super(MyModel_Chen, self).__init__()

        self.feature_map_3d = None
        self.feature_map_2d = None
        self.attention_map_2d = None
        self.attention_map_3d = None

        # Convolutional Layers
        self.conv_layer1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2, 3, 3))
        self.conv_layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2, 3, 3))

        # Reshape the Conv3D output to fit the subsequent Conv2D layer
        self.conv3d_to_2d = Conv3DTo2D()

        self.residual_conv_layer1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 1, 1))
        self.residual_conv_layer2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 1, 1))

        # Add a Conv2D layer
        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1))
        self.conv_layer5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))

        self.lrn1 = nn.LocalResponseNorm(16,1e-1)
        self.lrn2 = nn.LocalResponseNorm(32,1e-1)
        self.lrn3 = nn.LocalResponseNorm(64,1e-1)
        self.lrn4 = nn.LocalResponseNorm(128,1e-1)

        # Attention mechanism
        self.attention1 = SelfAttention3D(in_channels=16)
        self.attention2 = SelfAttention2D(in_channels=16)

        num = 0
        if dataset == 'IndianPines':
            num = 25472
        elif dataset == 'PaviaU':
            num = 13056
        elif dataset == 'PaviaC':
            num = 12928
        elif dataset == 'Botswana':
            num = 18432
        elif dataset == 'KSC':
            num = 22400

        # Fully Connected Layers
        self.dense_layer1 = nn.Linear(num, 256)

        # Output Layer
        self.output_layer = nn.Linear(256, output_units)

        # Dropout Layers
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):

        # 3D卷积层
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))

        self.feature_map_3d = x

        x_res = F.relu(self.residual_conv_layer1(x))
        x_res = self.residual_conv_layer2(x_res)
        x = x + x_res

        # 自注意力机制
        x, self.attention_map_3d = self.attention1(x)
        x = F.relu(self.lrn1(x))

        # 降维
        x = self.conv3d_to_2d(x)

        x, self.attention_map_2d = self.attention2(x)
        x = F.relu(self.lrn2(x))

        # 2D卷积层
        x = F.relu(self.conv_layer3(x))
        x = F.relu(self.conv_layer4(x))
        x = F.relu(self.lrn3(x))

        x = F.relu(self.conv_layer5(x))
        x = F.relu(self.conv_layer6(x))
        x = F.relu(self.lrn4(x))

        self.feature_map_2d = x

        # 扁平化
        x = torch.flatten(x, start_dim=1)

        # 全连接层
        x = F.relu(self.dropout(self.dense_layer1(x)))

        # 输出层
        x = self.output_layer(x)

        return x

def visualize_attention_map(attention_map, env_name='default', win='attention_map', colormap='Inferno'):
    """
    Visualizes the attention map as a heatmap using Visdom.
    Args:
        attention_map (numpy array): Attention map with shape (height, width).
        env_name (str): Name of the Visdom environment.
        win (str): Name of the window for the attention map visualization.
    """
    # 将注意力图进行归一化到 [0, 1] 范围
    normalized_attention_map = attention_map / attention_map.max()

    # 使用更明显的颜色映射，并增加对比度
    opts = dict(title=win, colormap=colormap, clim=[0, 1])

    vis = Visdom(env=env_name)
    vis.heatmap(
        X=normalized_attention_map,
        opts=opts,
        #win=win  # 使用新的窗口名
    )

class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1])))
        x = F.relu(self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0])))
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while (n > 1):
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


# 定义训练函数
def train_with_cross_validation(net, optimizer, criterion, data_loader, epoch, scheduler=None,
                                display_iter=100, device=torch.device('cpu'), display=None,
                                val_loader=None, supervision='full', n_splits=10, save_dir="saved_models"):
    kf = KFold(n_splits=n_splits, shuffle=True)

    val_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(data_loader.dataset), 1):
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        train_loader = DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=train_sampler)
        val_loader = DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=val_sampler)

        train(net, optimizer, criterion, train_loader, epoch, scheduler=scheduler,
              display_iter=display_iter, device=device, display=display,
              val_loader=val_loader, supervision=supervision)

        val_acc = val(net, val_loader, device=device, supervision=supervision)
        val_accuracies.append(val_acc)

        # 保存模型
        save_path = os.path.join('./checkpoints/' +
                                 camel_to_snake(str(net.__class__.__name__)) + '/' +
                                 data_loader.dataset.name + '/', f"fold_{fold}_model.pth")
        torch.save(net.state_dict(), save_path)
        print(f"Model for fold {fold} saved at: {save_path}")

    # 计算十次验证的性能指标的平均值
    avg_val_accuracy = np.mean(val_accuracies)
    print("Average validation accuracy over {} folds: {:.4f}".format(n_splits, avg_val_accuracy))


def train(net, optimizer, criterion, data_loader, hyperparams, epoch, dataset_name=None, scheduler=None,
          display_iter=25, device=torch.device('cpu'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 5 if epoch > 5 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    learning_rate_history = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                # target = target - 1
                loss = criterion(output, target)
            elif supervision == 'semi':
                outs = net(data)
                print(data)
                output, rec = outs
                # target = target - 1
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))

            loss.backward()

            if iter_ % 3500 == 0 or iter_ == 100:
                # 打印梯度信息
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            # 可视化梯度
                            grad_data = param.grad.view(-1)  # 将梯度展平成一维向量
                            visualize_gradients_visdom(grad_data, env_name=str(dataset_name) + ' GradientMap',
                                                       win='gradient, iter:' + str(iter_))
                        else:
                            print("Gradient is None for parameter:", param)

            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                              len(data), len(data) * len(data_loader),
                              100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                          }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                 })

            if iter_ % 3500 == 0 or iter_ == 100:

                visualize_attention_map(net.attention_map_3d, env_name=str(dataset_name) + ' AttentionMap',
                                        win='3D, iter:' + str(iter_))
                visualize_attention_map(net.attention_map_2d, env_name=str(dataset_name) + ' AttentionMap',
                                        win='2D, iter:' + str(iter_), colormap='Viridis')
                visualize_feature_maps_visdom_2D(net.feature_map_2d, env_name=str(dataset_name) + '2D Feature Map',
                                        win='2D feature, iter:' + str(iter_))
                visualize_feature_maps_visdom_3D(net.feature_map_3d, env_name=str(dataset_name) + '3D Feature Map',
                                        win='3D feature, iter:' + str(iter_))
                visualize_conv_weights_concat_visdom_3D(net, 'conv_layer2',env_name=str(dataset_name) + 'conv3d',
                                        win='3D conv3d, iter:' + str(iter_))
                visualize_conv_weights_concat_visdom_2D(net, 'conv_layer6',env_name=str(dataset_name) + 'conv2d',
                                        win='2D conv2d, iter:' + str(iter_))

            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e,
                       metric=abs(metric))

def visualize_gradients_visdom(grad_data, env_name='test', win='test'):
    vis = Visdom(env=env_name)

    # 将梯度数据移动到主机内存
    grad_data = grad_data.detach().cpu().numpy()

    # 将一维数组转换为二维数组
    if grad_data.ndim == 1:
        grad_data = grad_data.reshape(1, -1)

    # 获取梯度数据的形状
    num_samples, num_features = grad_data.shape

    # 取一半的数据
    half_num_features = num_features // 2
    half_grad_data = grad_data[:, :half_num_features]

    # 计算多少行和多少列可以展示所有梯度信息
    ncols = int(np.ceil(np.sqrt(half_grad_data.shape[1])))
    nrows = int(np.ceil(half_grad_data.shape[1] / ncols))

    # 将梯度数据扁平化为一维数组，并填充为能够填满所有格子的形状
    padded_grad_data = np.zeros((nrows * ncols,))
    padded_grad_data[:half_grad_data.shape[1]] = half_grad_data.flatten()

    # 调整对比度
    padded_grad_data = (padded_grad_data - padded_grad_data.min()) / (
                padded_grad_data.max() - padded_grad_data.min()) * 255

    # 计算图的数量，最多输出10张图
    num_images = min(10, nrows * ncols)

    # 选择子集进行可视化
    subset_indices = np.linspace(0, padded_grad_data.shape[0] - 1, num=num_images, dtype=int)
    subset_data = padded_grad_data[subset_indices]

    # 将一维数组重新转换为二维数组
    subset_ncols = min(10, ncols)
    subset_nrows = int(np.ceil(num_images / subset_ncols))
    subset_data = subset_data.reshape(subset_nrows, subset_ncols)

    # 可视化梯度数据
    opts = dict(title=win, width=250, height=250, colormap='cividis')
    vis.images(subset_data, opts=opts)



def visualize_feature_maps_visdom_2D(feature_map_2d, env_name='test', win='test', ncols=1):
    vis = Visdom(env=env_name)

    # 将特征图移动到主机内存
    feature_map_2d = feature_map_2d.detach().cpu()

    # 调整形状以便在通道维度上堆叠特征图
    feature_map_stacked = feature_map_2d.permute(0, 2, 3, 1)

    # 每隔一个特征图进行可视化
    feature_map_sampled = feature_map_stacked[:, ::125]

    # 调整对比度
    feature_map_sampled = (feature_map_sampled - feature_map_sampled.min()) / (
                feature_map_sampled.max() - feature_map_sampled.min()) * 255

    # 转换为 NumPy 数组
    feature_map_np = feature_map_sampled.numpy()

    # 可视化特征图
    opts = dict(title=win, width=1000, height=250, colormap='cividis',
                nrow=ncols)
    vis.images(feature_map_np, opts=opts)

def visualize_feature_maps_visdom_3D(feature_map_3d, env_name='test', win='test', ncols=8):
    vis = Visdom(env=env_name)

    # 获取特征图的形状
    batch_size, channels, depth, height, width = feature_map_3d.shape

    # 将特征图重新排列为 4D 张量
    feature_map_4d = feature_map_3d.permute(0, 2, 3, 4, 1)  # 将样本维度移到最前面

    # 将特征图的每个样本拼接在一起
    feature_map_concatenated = feature_map_4d.reshape(-1, height, width, channels)

    # 去除梯度并转换为浮点数类型
    feature_map_concatenated = feature_map_concatenated.detach().cpu().float()

    # 归一化特征图到 [0, 255] 范围
    feature_map_concatenated = (feature_map_concatenated - feature_map_concatenated.min()) / (
            feature_map_concatenated.max() - feature_map_concatenated.min()) * 255

    # 转换为整型类型
    feature_map_concatenated = feature_map_concatenated.to(torch.uint8)

    # 每隔一个特征图进行可视化
    feature_map_concatenated_sampled = feature_map_concatenated[::100]

    # 转换为 NumPy 数组
    feature_map_concatenated_np = feature_map_concatenated_sampled.numpy()

    # 可视化特征图
    opts = dict(title=win, width=1000, height=250, colormap='cividis',
                nrow=ncols)
    vis.images(feature_map_concatenated_np, opts=opts)

def visualize_conv_weights_concat_visdom_3D(model, layer_name, ncols=4, env_name='test', win='test'):
    vis = Visdom(env=env_name)
    # 获取指定名称的卷积层
    conv_layer = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, torch.nn.Conv3d):
            conv_layer = module
            break

    if conv_layer is None:
        print("No convolutional layer found with name:", layer_name)
        return

    # 获取卷积层的权重
    conv_weights = conv_layer.weight.data.cpu().numpy()
    #print("Convolutional weights shape:", conv_weights.shape)  # 调试信息

    # 获取输出通道数、输入通道数、核深度、核高度和核宽度
    output_channels, input_channels, kernel_depth, kernel_height, kernel_width = conv_weights.shape

    # 计算行数
    nrows = int(np.ceil(output_channels / ncols))

    # 创建拼接后的图像
    concatenated_weights = np.zeros((nrows * kernel_height * kernel_depth, ncols * kernel_width * input_channels))

    # 将每个卷积核图像按照多行多列的方式拼接在一起
    for i in range(output_channels):
        row_idx = i // ncols
        col_idx = i % ncols
        start_row = row_idx * kernel_height * kernel_depth
        end_row = start_row + kernel_height * kernel_depth
        start_col = col_idx * kernel_width * input_channels
        end_col = start_col + kernel_width * input_channels
        concatenated_weights[start_row:end_row, start_col:end_col] = np.reshape(conv_weights[i], (
        kernel_height * kernel_depth, kernel_width * input_channels))

    opts = dict(title=win, colormap='Inferno', clim=[np.min(concatenated_weights), np.max(concatenated_weights)], width=250, height=250)
    vis.image(concatenated_weights, opts=opts)

def visualize_conv_weights_concat_visdom_2D(model, layer_name, ncols=4, env_name='test',
                                            win = 'test'):
    vis = Visdom(env=env_name)
    # 获取指定名称的卷积层
    conv_layer = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, torch.nn.Conv2d):
            conv_layer = module
            break

    if conv_layer is None:
        print("No convolutional layer found with name:", layer_name)
        return

    # 获取卷积层的权重
    conv_weights = conv_layer.weight.data.cpu().numpy()

    #print("Convolutional weights shape:", conv_weights.shape)

    # 获取通道数和核大小
    channels, kernel_height, kernel_width = conv_weights.shape[1:]

    # 计算行列数
    nrows = int(np.ceil(channels / ncols))

    # 创建拼接后的图像
    concatenated_weights = np.zeros((nrows * kernel_height, ncols * kernel_width))

    # 将每个卷积核图像按照多行多列的方式拼接在一起
    for i in range(channels):
        row_idx = i // ncols
        col_idx = i % ncols
        start_row = row_idx * kernel_height
        end_row = start_row + kernel_height
        start_col = col_idx * kernel_width
        end_col = start_col + kernel_width
        concatenated_weights[start_row:end_row, start_col:end_col] = conv_weights[0, i]

    # 可视化拼接后的图像
    opts = dict(title=win, colormap='Inferno', clim=[0, 1],width=250, height=250)
    vis.image(concatenated_weights,
              opts=opts)

def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('run') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
    else:
        filename = str('run')
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out

    return probs


def val(net, data_loader, device='cpu', supervision='full'):
    # TODO : fix me using metrics()
    net.eval()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # target = target - 1
            for pred, out in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
