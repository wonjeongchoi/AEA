# -*- coding: utf-8 -*-
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.helpers import checkpoint_seq
from ttab.configs.datasets import dataset_defaults

__all__ = ["resnet", "resnet50"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = in_planes + out_planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = (
            (in_planes + 2 * out_planes) * in_planes / (2 * in_planes + out_planes)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)
        return out


class ResNetBase(nn.Module):
    def _init_conv(self, module):
        out_channels, _, kernel_size0, kernel_size1 = module.weight.size()
        n = kernel_size0 * kernel_size1 * out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _init_bn(self, module):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

    def _init_fc(self, module):
        module.weight.data.normal_(mean=0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

    def _weight_initialization(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self._init_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                self._init_bn(module)
            elif isinstance(module, nn.Linear):
                self._init_fc(module)

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.inplanes,
                                planes * block_fn.expansion,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        (
                            "bn",
                            norm2d(
                                group_norm_num_groups,
                                planes=planes * block_fn.expansion,
                            ),
                        ),
                    ]
                )
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )
        return nn.Sequential(*layers)


class ResNetImagenet(ResNetBase):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        split_point: str = "layer4",
        group_norm_num_groups: int = None,
        grad_checkpoint: bool = False,
    ):
        super(ResNetImagenet, self).__init__()
        self.num_classes = num_classes
        if split_point not in ["layer3", "layer4", None]:
            raise ValueError(f"invalid split position={split_point}.")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint

        # define model param.
        self.depth = depth
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[depth]["block"]
        block_nums = model_params[depth]["layers"]

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(
            in_features=512 * block_fn.expansion,
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Forward function without classifier. Use gradient checkpointing to save memory."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer2, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
            if self.split_point in ["layer4", None]:
                x = checkpoint_seq(self.layer4, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            if self.split_point in ["layer4", None]:
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """Forward function for classifier. Use gridient checkpointing to save memory."""
        if self.split_point == "layer3":
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer4, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            else:
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResNetCifar(ResNetBase):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        split_point: str = "layer3",
        group_norm_num_groups: int = None,
        grad_checkpoint: bool = False,
    ):
        super(ResNetCifar, self).__init__()
        self.num_classes = num_classes
        if split_point not in ["layer2", "layer3", None]:
            raise ValueError(f"invalid split position={split_point}.")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint

        # define model.
        self.depth = depth
        if depth % 6 != 2:
            raise ValueError("depth must be 6n + 2:", depth)
        block_nums = (depth - 2) // 6
        block_fn = Bottleneck if depth >= 44 else BasicBlock
        self.block_nums = block_nums
        self.block_fn_name = "Bottleneck" if depth >= 44 else "BasicBlock"

        # define layers.
        self.inplanes = int(16)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=int(16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * block_fn.expansion),
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Forward function without classifier. Use gradient checkpointing to save memory."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer2, x, preserve_rng_state=True)
            if self.split_point in ["layer3", None]:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            # original version
            x = self.layer1(x)
            x = self.layer2(x)
            if self.split_point in ["layer3", None]:
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

            # # my version (to visualize features)
            # x = self.layer1(x)
            # l1_stats = get_style_stats_merged(x)
            # x = self.layer2(x)
            # l2_stats = get_style_stats_merged(x)
            # if self.split_point in ["layer3", None]:
            #     x = self.layer3(x)
            #     l3_stats = get_style_stats_merged(x)
            #     x = self.avgpool(x)
            #     x = x.view(x.size(0), -1)
            #
            # import numpy as np
            # import os
            # import time
            # save_dir = './logs/resnet26/noadapt_debug'
            # file_name = 'stats_array_' + time.strftime("%Y%m%d") + '.npz'  # time.strftime("%Y%m%d_%H%M%S")
            # full_file_name = os.path.join(save_dir, file_name)
            #
            # if os.path.isfile(full_file_name):
            #     saved_array = np.load(full_file_name)
            #     l1_stats = np.concatenate((saved_array['l1_stats'], np.array(l1_stats.cpu())), axis=0)
            #     l2_stats = np.concatenate((saved_array['l2_stats'], np.array(l2_stats.cpu())), axis=0)
            #     l3_stats = np.concatenate((saved_array['l3_stats'], np.array(l3_stats.cpu())), axis=0)
            #     np.savez(full_file_name,
            #              l1_stats=l1_stats,
            #              l2_stats=l2_stats,
            #              l3_stats=l3_stats
            #              )
            #     saved_array.close()
            #
            # else:
            #     np.savez(full_file_name,
            #              l1_stats=np.array(l1_stats.cpu()),
            #              l2_stats=np.array(l2_stats.cpu()),
            #              l3_stats=np.array(l3_stats.cpu())
            #              )

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """Forward function for classifier. Use gridient checkpointing to save memory."""
        if self.split_point == "layer2":
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            else:
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResNetMNIST(ResNetBase):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        in_dim: int,
        split_point: str = "layer4",
        group_norm_num_groups: int = None,
    ):
        super(ResNetMNIST, self).__init__()
        self.num_classes = num_classes
        if split_point not in ["layer3", "layer4", None]:
            raise ValueError(f"invalid split position={split_point}.")
        self.split_point = split_point

        # define model.
        self.depth = depth
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[depth]["block"]
        block_nums = model_params[depth]["layers"]

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.classifier = nn.Linear(
            in_features=int(512 * block_fn.expansion),
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Forward function without classifier. Use gradient checkpointing to save memory."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.split_point in ["layer4", None]:
            x = self.layer4(x)
            x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """Forward function for classifier. Use gridient checkpointing to save memory."""
        if self.split_point == "layer3":
            x = self.layer4(x)
            x = x.view(x.size(0), -1)

        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def resnet(
    dataset: str,
    depth: int,
    split_point: str = None,
    group_norm_num_groups: int = None,
    grad_checkpoint: bool = False,
):
    num_classes = dataset_defaults[dataset]["statistics"]["n_classes"]
    if "mnist" in dataset:
        in_dim = 3
        if dataset == "mnist":
            in_dim = 1
        return ResNetMNIST(
            num_classes, depth, in_dim, split_point, group_norm_num_groups
        )
    elif "yearbook" in dataset:
        return ResNetMNIST(
            num_classes, depth, in_dim=3, split_point=split_point, group_norm_num_groups=group_norm_num_groups
        )
    elif "cifar" in dataset:
        return ResNetCifar(
            num_classes, depth, split_point, group_norm_num_groups, grad_checkpoint
        )
    else:
        if depth==50:
            return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        else:
            raise NotImplementedError

        # return ResNetImagenet(
        #     num_classes, depth, split_point, group_norm_num_groups, grad_checkpoint
        # )


def get_style_stats_merged(x):
    B, C, H, W = x.size()

    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + 1e-6).sqrt()
    mu, sig = mu.detach(), sig.detach()
    return torch.cat([mu, sig], dim=1)



###  another version of ResNet
def conv3x3_v2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_v2(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_only=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        if return_feature:
            feature = x
        x = self.fc(x)

        if return_feature:
            if return_feature_only:
                return feature
            else:
                return x, feature

        else:
            return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck2, [3, 4, 6, 3], **kwargs)