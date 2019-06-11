# 20190606 Used to Calculate the number of model parameters.

import numpy as np
import pickle
import yaml

import torch
import torch.nn as nn




class AlexNetS(nn.Module):
	def __init__(self):
		super(AlexNetS, self).__init__()
		def conv_bn(inp, oup, kernel, stride, pad, groups):
			return nn.Sequential(
				nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=kernel, stride=stride, padding=pad, groups=groups, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
			)
		self.classifier = nn.Sequential(
			nn.Linear(192 * 8 * 8, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Linear(256, num_of_classes, bias=True),
		)
		self.features = nn.Sequential(
			conv_bn(inp=3,oup=48,kernel=3,stride=1,pad=1,groups=3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			conv_bn(inp=48,oup=96,kernel=3,stride=1,pad=1,groups=3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			conv_bn(inp=96,oup=192,kernel=3,stride=1,pad=1,groups=3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			conv_bn(inp=192,oup=192,kernel=3,stride=1,pad=1,groups=3),
			conv_bn(inp=192,oup=192,kernel=3,stride=1,pad=1,groups=3), # For (100,128) input, the feature map size after this layer is (24,31)
			nn.MaxPool2d(kernel_size=2, stride=2), #If this is used, then feature map size =(11,15)
			#nn.AdaptiveAvgPool2d((1,1)),
		)
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_of_classes=10
model = AlexNetS()
print('The number of parameters is %d' % count_parameters(model))

