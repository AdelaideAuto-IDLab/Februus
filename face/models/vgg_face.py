# -*- coding: utf-8 -*-

# We acknowledge the following repo from where we adopt the VGG-Face model
# https://github.com/prlz77/vgg-face.pytorch
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_1_2 = nn.ReLU()
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.relu_2_2 = nn.ReLU()
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.relu_3_2 = nn.ReLU()
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.relu_3_3 = nn.ReLU()
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu_4_2 = nn.ReLU()
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu_4_3 = nn.ReLU()
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu_5_2 = nn.ReLU()
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu_5_3 = nn.ReLU()
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = self.relu_1_1(self.conv_1_1(x))
        x = self.relu_1_2(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu_2_1(self.conv_2_1(x))
        x = self.relu_2_2(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu_3_1(self.conv_3_1(x))
        x = self.relu_3_2(self.conv_3_2(x))
        x = self.relu_3_3(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu_4_1(self.conv_4_1(x))
        x = self.relu_4_2(self.conv_4_2(x))
        x = self.relu_4_3(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu_5_1(self.conv_5_1(x))
        x = self.relu_5_2(self.conv_5_2(x))
        x = self.relu_5_3(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.relu6(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.relu7(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

