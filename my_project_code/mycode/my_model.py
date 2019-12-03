import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
# using partseg as segmentation, cls as feature extraction and 2 FC layers at the end of the net.
class PointNet2PartSeg(nn.Module): # TODO part segmentation tasks
    def __init__(self, num_classes):
        super(PointNet2PartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, feat

class Feature_extract(nn.Module):
    def __init__(self):
        super(Feature_extract, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 0+3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz, norm_plt, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.size()
        l0_xyz = xyz
        l0_points = norm_plt
        l1_xyz, l1_points = self.sa1(l0_xyz, norm_plt)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        return x
'''
class fullyconnect_concat(nn.Module):
    def __init__(self, num_part):    # the size of FC layer can be changed depend on the effect.
        super(fullyconnect_concat, self).__init__()
        self.layer1 = nn.Linear(1024 * num_part, 128)   # num of output can be changed depend on the effect.
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, 16)

    def forward(self, in_feat):            # in_feat is the concatenation of Feature_extract.
        # x = nn.MaxPool2d(in_feat, (2, 2), 1)  # can be change
        x = nn.ReLU(self.layer1(in_feat))
        x = nn.Dropout(x, p=0.5)        # can be change
        x = nn.ReLU(self.layer2(x))
        x = nn.Dropout(x, p=0.5)
        return nn.Softmax(nn.ReLU(self.layer3(x)), dim=1)
'''
class FC_pooling(nn.Module):    # preprocess the data with element-wise max operation.
    def __init__(self):    # the size of FC layer can be changed depend on the effect.
        super(FC_pooling, self).__init__()
        self.layer1 = nn.Linear(1024, 256)   # num of output can be changed depend on the effect.
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 16)

    def forward(self, in_feat):            # in_feat is the concatenation of Feature_extract.
        # x = nn.MaxPool2d(in_feat, (2, 2), 1)  # can be change
        x = nn.ReLU(self.layer1(in_feat))
        x = nn.Dropout(x, p=0.5)        # can be change
        x = nn.ReLU(self.layer2(x))
        x = nn.Dropout(x, p=0.5)
        return nn.Softmax(nn.ReLU(self.layer3(x)), dim=1)