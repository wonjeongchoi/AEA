# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ttab.loads.models.resnet import (
    ResNetCifar,
    ResNetImagenet,
    ResNetMNIST,
    ViewFlatten,
)
from ttab.loads.models.wideresnet import WideResNet

import ttab.configs.utils as configs_utils
import ttab.loads.define_dataset as define_dataset
from sklearn import linear_model

"""optimization dynamics"""


def define_optimizer(meta_conf, params, lr=1e-3):
    """Set up optimizer for adaptation."""
    weight_decay = meta_conf.weight_decay if hasattr(meta_conf, "weight_decay") else 0

    if not hasattr(meta_conf, "optimizer") or meta_conf.optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=meta_conf.momentum if hasattr(meta_conf, "momentum") else 0.9,
            dampening=meta_conf.dampening if hasattr(meta_conf, "dampening") else 0,
            weight_decay=weight_decay,
            nesterov=meta_conf.nesterov if hasattr(meta_conf, "nesterov") else True,
        )
    elif meta_conf.optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(meta_conf.beta if hasattr(meta_conf, "beta") else 0.9, 0.999),
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError


def lr_scheduler(optimizer, iter_ratio, gamma=10, power=0.75):
    decay = (1 + gamma * iter_ratio) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
    return optimizer


class SAM(torch.optim.Optimizer):
    """
    SAM is an optimizer proposed to seek parameters that lie in neighborhoods having uniformly low loss.

    Sharpness-Aware Minimization for Efficiently Improving Generalization
    https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


"""method-wise modification on model structure"""

# for bn_adapt
def modified_bn_forward(self, input):
    """
    Leverage the statistics already computed on the seen data as a prior and infer the test statistics for each test batch as a weighted sum of
    prior statistics and estimated statistics on the current batch.

    Improving robustness against common corruptions by covariate shift adaptation
    https://arxiv.org/abs/2006.16971
    """
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(
        input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps
    )


class shared_ext_from_layer4(nn.Module):
    """
    Select all layers before layer4 and layer4 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetImagenet):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, ResNetMNIST):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer3(nn.Module):
    """
    Select all layers before layer3 and layer3 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer2(nn.Module):
    """
    Select all layers before layer2 and layer2 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


def head_from_classifier(model, dim_out):
    """Select the last classifier layer in ResNets as head."""
    # Self-supervised task used in TTT is rotation prediction. Thus the out_features = 4.
    head = nn.Linear(
        in_features=model.classifier.in_features, out_features=dim_out, bias=True
    )
    return head


def head_from_last_layer1(model, dim_out):
    """
    Select the layer 3 or 4 and the following classifier layer as head.

    Only used for ResNets.
    """
    if isinstance(model, ResNetCifar):
        head = copy.deepcopy([model.layer3, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetImagenet):
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetMNIST):
        head = copy.deepcopy([model.layer4])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, WideResNet):
        head = copy.deepcopy([model.layer3, model.bn1, model.relu, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, models.ResNet):
        # for torchvision.models.resnet50
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.fc.in_features, dim_out, bias=False))
    else:
        raise NotImplementedError

    return nn.Sequential(*head)


class ExtractorHead(nn.Module):
    """
    Combine the extractor and the head together in ResNets.
    """

    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))

    def make_train(self):
        self.ext.make_train()
        self.head.train()

    def make_eval(self):
        self.ext.make_eval()
        self.head.eval()


class VitExtractor(nn.Module):
    """
    Combine the extractor and the head together in ViTs.
    """

    def __init__(self, model):
        super(VitExtractor, self).__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        x = self.model.forward_features(x)
        if self.model.global_pool:
            x = (
                x[:, self.model.num_prefix_tokens :].mean(dim=1)
                if self.model.global_pool == "avg"
                else x[:, 0]
            )
        x = self.model.fc_norm(x)
        return x

    def _select_layers(self):
        layers = []
        for named_module, module in self.model.named_children():
            if not module == self.model.get_classifier():
                layers.append(module)
        return layers

    def make_train(self):
        for layer in self.layers:
            layer.train()

    def make_eval(self):
        for layer in self.layers:
            layer.eval()


# for ttt++
class FeatureQueue:
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr : self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue
        else:
            return None


# for note
class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k = k
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(
                x, dim=[0, 2, 3], keepdim=True, unbiased=True
            )
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2, 3], keepdim=True, unbiased=True
                )
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)

        if h * w <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )

            sigma2_adj = F.relu(sigma2_adj)  # non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm1d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2], keepdim=True, unbiased=True
                )
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        if l <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b

        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l)  ##
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n


"""Auxiliary tasks"""

# rotation prediction task
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label, device, generator=None):
    if label == "rand":
        labels = torch.randint(
            4, (len(batch),), generator=generator, dtype=torch.long
        ).to(device)
    elif label == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        ).to(device)
        batch = batch.repeat((4, 1, 1, 1))

    return rotate_batch_with_labels(batch, labels), labels


"""loss-related functions."""


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy_decompose(x):
    """Decomposed Entropy of softmax distribution from logits."""
    weighted_energys = -(x.softmax(1) * x).sum(1)
    neg_free_energys = -1 * energy(x)
    return weighted_energys, neg_free_energys


def teacher_student_softmax_entropy(
    x: torch.Tensor, x_ema: torch.Tensor
) -> torch.Tensor:
    """Cross entropy between the teacher and student predictions."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def entropy(input):
    bs = input.size(0)
    ent = -input * torch.log(input + 1e-5)
    ent = torch.sum(ent, dim=1)
    return ent

#################### Energy-related ###############################
def energy(input, temp=1.0):
    bs = input.size(0)
    energy = -temp * torch.logsumexp(input/temp, dim=1)
    return energy

def adaptive_energy(input, temp=1.0):
    bs = input.size(0)
    weight = 1/softmax_entropy(input).exp()
    #weight = -1 * softmax_entropy(input)
    weight = weight.detach()
    energy_values = energy(input, temp=temp)
    return weight * energy_values

# def energy_alignment_with_sample(input, temp=1.0):
#     energy = -temp * torch.logsumexp(input / temp, dim=1)
#     diff = -1
#     loss = nn.ReLU()(diff)
#     return loss

def oracle_energy_alignment(input, src_energy, temp=1.0):
    energy = -temp * torch.logsumexp(input / temp, dim=1)
    tar_energy = energy.detach().mean(0)
    diff = energy - src_energy
    loss = nn.ReLU()(diff)
    return loss, tar_energy

def sample_selective_energy_alignment(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()

    src_energy_approx = torch.chunk(values, num_chunk)[0].mean()
    diff = energy - src_energy_approx
    loss = nn.ReLU()(diff)
    return loss, src_energy_approx, tar_energy

def sample_selective_softplus_energy_alignment(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()

    src_energy_approx = torch.chunk(values, num_chunk)[0].mean()
    diff = energy - src_energy_approx
    loss = nn.Softplus()(diff)
    return loss


def energy_distribution_matching(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()
    tar_mean, tar_std = energy.mean(), energy.std()

    src_energy = torch.chunk(values, num_chunk)[0]
    src_mean, src_std = src_energy.mean(), src_energy.std()

    diff = energy - src_mean
    loss = nn.Softplus()(diff).mean()
    # loss = torch.abs(tar_mean-src_mean)
    return loss

############ Energy-adjusted Entropy Minimization (EEM) ############
def energy_adj_ent_min(input, energy_weight, temp=1.0):
    bs = input.size(0)
    ent = softmax_entropy(input)
    eng = energy(input, temp=temp)
    return ent + energy_weight * eng

# def adaptive_energy_adj_ent_min(input, energy_weight, temp=1.0):
#     bs = input.size(0)
#     ent = softmax_entropy(input)
#     eng = adaptive_energy(input, temp=temp)
#     return ent + energy_weight * eng
#
# def adaptive_energy_adj_ent_min2(input, energy_weight, thr, temp=1.0):
#     bs = input.size(0)
#     ent = softmax_entropy(input)
#
#     # filter unreliable samples
#     filter_ids_1 = torch.where(ent < thr)
#     # ids1 = filter_ids_1
#     # ids2 = torch.where(ids1[0] > -0.1)
#     # ent = ent[filter_ids_1]
#
#     coeff = energy_weight / (
#         torch.exp(ent.clone().detach() - thr)
#     )
#     # ent = ent.mul(coeff)
#
#     eng = energy(input, temp=temp)
#     eng = coeff * eng
#     eng = eng[filter_ids_1]
#     loss = ent.mean(0) + eng.mean(0)
#
#     # weight = torch.zeros_like(ent)
#     # weight[filter_ids_1] = coeff[filter_ids_1]
#     # loss = ent + weight * eng
#     return loss
#
# def csa_energy_adj_ent_min(input, energy_weight, conf_thr, version, temp=1.0):
#     """Confidence-based Filtering & Energy-adjusted EM loss"""
#     if version == 0:
#         ################### version 1 ####################################
#         # compute eem loss
#         ent = softmax_entropy(input)
#         eng = energy(input, temp=temp)
#         loss = ent + energy_weight * eng
#
#         # confidence-based filtering
#         prob_all, indices_all = input.softmax(-1).max(-1)
#         filter_ids = torch.where(prob_all >= conf_thr)
#         return loss[filter_ids]
#     elif version == 1:
#         ############################ version 2 ###########################
#         # compute entropy and energy
#         ent = softmax_entropy(input)
#         eng = energy(input, temp=temp)
#
#         # confidence-based filtering
#         prob_all, indices_all = input.softmax(-1).max(-1)
#         filter_ids = torch.where(prob_all >= conf_thr)
#
#         weight = torch.zeros_like(ent)
#         weight[filter_ids] = energy_weight
#         loss = ent + weight * eng
#         return loss
#
# def esa_energy_adj_ent_min(input, energy_weight, ent_thr, version, temp=1.0):
#     """Entropy-based Filtering & Energy-adjusted EM loss"""
#     if version == 0:
#         # compute eem loss
#         ent = softmax_entropy(input)
#         eng = energy(input, temp=temp)
#         loss = ent + energy_weight * eng
#
#         # entropy-based filtering
#         filter_ids = torch.where(ent < ent_thr)
#         return loss[filter_ids]
#
#     elif version == 1:
#         # compute entropy and energy
#         ent = softmax_entropy(input)
#         eng = energy(input, temp=temp)
#
#         # entropy-based filtering
#         filter_ids = torch.where(ent < ent_thr)
#
#         weight = torch.zeros_like(ent)
#         weight[filter_ids] = energy_weight
#         loss = ent + weight * eng
#         return loss

########################################################################

def logit_similarity(input, cls_weight, thr=0.):
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)

    # simple pseudo-labeling
    prob_all, indices_all = input.softmax(-1).max(-1)
    virtual_logits = classwise_virtual_logits[indices_all]
    conf_indices = prob_all>=thr
    loss = 1 - F.cosine_similarity(input[conf_indices], virtual_logits[conf_indices], dim=-1)
    return loss

def weighted_lcs(input, cls_weight, thr=0.):
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)

    # simple pseudo-labeling
    prob_all, indices_all = input.softmax(-1).max(-1)
    virtual_logits = classwise_virtual_logits[indices_all]
    conf_indices = prob_all>=thr
    loss = 1 - F.cosine_similarity(input, virtual_logits, dim=-1)
    loss = (prob_all.detach() - thr).exp() * loss
    loss = loss[conf_indices]

    # option = 0
    # if option == 0:
    #     loss = prob_all.detach().exp() * loss
    # else:
    #     loss = (prob_all.detach()-thr).exp() * loss
    # loss = loss[conf_indices]

    return loss

# def esa_logit_similarity(input, cls_weight, thr=0.):
#     classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
#     #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)
#
#     # simple pseudo-labeling
#     ent = softmax_entropy(input)
#     prob_all, indices_all = input.softmax(-1).max(-1)
#     virtual_logits = classwise_virtual_logits[indices_all]
#     filtered_indices = ent < thr
#     loss = 1 - F.cosine_similarity(input[filtered_indices], virtual_logits[filtered_indices], dim=-1)
#     return loss










# class EALoss:  # Energy Alignment Loss
#     def __init__(
#         self, device
#     ):
#         super(EALoss, self).__init__()
#         self.saved_energy_tensor = torch.tensor([float("Inf")], device=device)
#
#     def energy_alignment_loss(self, input, weight, temp=1.0):
#         bs = input.size(0)
#         mse = nn.MSELoss()
#
#         ### 1)
#         # energy_values = energy(input, temp=temp)
#         # source_energy = -15. * torch.ones_like(energy_values)
#         # loss = mse(energy_values, source_energy)
#
#         ### 2)
#         # energy_values = energy(input, temp=temp)
#         # energy_mean = energy_values.mean(0)
#         # loss = (energy_mean-(-15)).pow(2)
#
#         ### 3
#         # energy_values = energy(input, temp=temp)
#         # min_engy_idx = energy_values.min(0).indices
#         # min_engy_input = input[min_engy_idx].view(1,-1).detach()
#         # source_engy_approx = energy(min_engy_input, temp=temp)
#         # loss = mse(energy_values, source_engy_approx.expand(bs,))
#
#         ### 3-1
#         # energy_values = energy(input, temp=temp)
#         # min_engy_idx = energy_values.min(0).indices
#         # min_engy_input = input[min_engy_idx].view(1,-1).detach()
#         # source_engy_approx = energy(min_engy_input, temp=temp)
#         #
#         # if source_engy_approx.data < self.saved_energy_tensor.data:
#         #     self.saved_energy_tensor = source_engy_approx
#         #
#         # loss = mse(energy_values, self.saved_energy_tensor.expand(bs,))
#
#         ### 4 각 batch에서 max conf인 샘플의 energy 사용
#         # energy_values = energy(input, temp=temp)
#         # max_conf_idx = input.softmax(1).max(1).values.max(0).indices
#         # max_conf_input = input[max_conf_idx].view(1, -1).detach()
#         # source_engy_approx = energy(max_conf_input, temp=temp)
#         # loss = mse(energy_values, source_engy_approx.expand(bs, ))
#
#         ### 4-1 max conf인 샘플의 energy 저장해뒀다가 사용
#         # energy_values = energy(input, temp=temp)
#         # max_conf_idx = input.softmax(1).max(1).values.max(0).indices
#         # max_conf_input = input[max_conf_idx].view(1, -1).detach()
#         # source_engy_approx = energy(max_conf_input, temp=temp)
#         #
#         # if source_engy_approx.data < self.saved_energy_tensor.data:
#         #     self.saved_energy_tensor = source_engy_approx
#         #
#         # loss = mse(energy_values, self.saved_energy_tensor.expand(bs,))
#
#
#         return loss


def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4.0 * d**2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super().__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b



#############################################################################################
def make_src_loader(config):
    # make source data loader
    _, scenario = configs_utils.config_hparams(config=config)
    src_data_cls = define_dataset.ConstructAuxiliaryDataset(config=config)
    src_dataset, src_dataloader = src_data_cls.construct_src_dataset(scenario=scenario, data_size=1000)
    return src_dataloader

def make_src_train_loader(config):
    # make source data loader
    _, scenario = configs_utils.config_hparams(config=config)
    src_data_cls = define_dataset.ConstructAuxiliaryDataset(config=config)
    src_dataset, src_dataloader = src_data_cls.construct_src_train_dataset(scenario=scenario, data_size=10000)
    return src_dataloader

def make_trg_loader(config):
    # make target data loader
    _, scenario = configs_utils.config_hparams(config=config)
    # trg_data_cls = define_dataset.ConstructTestDataset(config=config)
    # test_dataloader = trg_data_cls.construct_test_loader(scenario=scenario)

    test_data_cls = define_dataset.ConstructAuxiliaryDataset(config=config)
    test_loader = test_data_cls.construct_auxiliary_loader(scenario=scenario)

    return test_loader

def src_prediction(config, model, src_loader):
    src_energies = []

    model.eval()
    with torch.no_grad():
        for step, epoch, batch in src_loader.iterator(
                batch_size=config.batch_size,  # apply the batch-wise or sample-wise setting.
                shuffle=False,  # we will apply shuffle operation in preparing dataset.
                repeat=False,
                ref_num_data=None,
                num_workers=config.num_workers
                if hasattr(config, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=False,
        ):
            yhat = model(batch._x)
            eng = energy(yhat)
            src_energies.append(eng)
    model.train()

    src_energies = torch.concat(src_energies, dim=0)
    mean_energy = src_energies.mean(0)
    return mean_energy

## from https://github.com/thuml/TransCal.git
def get_weight(source_train_feature, target_feature, source_val_feature):
    """
    :param source_train_feature: shape [n_tr, d], features from training set
    :param target_feature: shape [n_t, d], features from test set
    :param source_val_feature: shape [n_v, d], features from validation set

    :return:
    """
    # print("-"*30 + "get_weight" + '-'*30)
    n_tr, d = source_train_feature.shape
    n_t, _d = target_feature.shape
    # n_v, _d = source_val_feature.shape
    # print("n_tr: ", n_tr, "n_v: ", n_v, "n_t: ", n_t, "d: ", d)

    if n_tr < n_t:
        sample_index = np.random.choice(n_tr,  n_t, replace=True)
        source_train_feature = source_train_feature[sample_index]
        sample_num = n_t
    elif n_tr > n_t:
        sample_index = np.random.choice(n_t, n_tr, replace=True)
        target_feature = target_feature[sample_index]
        sample_num = n_tr

    combine_feature = np.concatenate((source_train_feature, target_feature))
    combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
    domain_classifier = linear_model.LogisticRegression()
    domain_classifier.fit(combine_feature, combine_label)
    domain_out = domain_classifier.predict_proba(source_val_feature)
    weight = domain_out[:, :1] / domain_out[:, 1:]
    return weight


def feature_vis(meta_conf, model, num_past_batches, loss):
    # for feature visualization
    import numpy as np
    import os
    import time
    save_dir = f'./logs/resnet26/{meta_conf.job_name}'
    # file_name = 'energy_anal_' + time.strftime("%Y%m%d") + '.npz'  # time.strftime("%Y%m%d_%H%M%S")
    file_name = f'feature_vis_{loss}_seed{meta_conf.seed}_{meta_conf.data_names}.npz'
    full_file_name = os.path.join(save_dir, file_name)

    if (num_past_batches == 0) or (num_past_batches % 10 == 0):  # os.path.isfile(full_file_name):
        ## 1) source domain
        # self._src_loader = adaptation_utils.make_src_loader(self._meta_conf)
        src_loader = make_src_train_loader(meta_conf)
        src_energies = []
        src_gts = []
        src_logits = []
        src_features = []
        model.eval()
        with torch.no_grad():
            for step, epoch, batch_s in src_loader.iterator(
                    batch_size=meta_conf.batch_size,  # apply the batch-wise or sample-wise setting.
                    shuffle=False,  # we will apply shuffle operation in preparing dataset.
                    repeat=False,
                    ref_num_data=None,
                    num_workers=meta_conf.num_workers
                    if hasattr(meta_conf, "num_workers")
                    else 2,
                    pin_memory=True,
                    drop_last=False,
            ):

                # for feature visualization
                feat_s = model.forward_features(batch_s._x)
                yhat_s = model.forward_head(feat_s)

                eng_s = energy(yhat_s)

                src_logits.append(yhat_s)
                src_gts.append(batch_s._y)
                src_energies.append(eng_s)
                src_features.append(feat_s)
        model.train()
        src_logits = torch.concat(src_logits, dim=0)
        src_gts = torch.concat(src_gts, dim=0)
        src_energies = torch.concat(src_energies, dim=0)
        src_features = torch.concat(src_features, dim=0)

        ## 2) target domain
        trg_loader = make_trg_loader(meta_conf)
        trg_energies = []
        trg_gts = []
        trg_logits = []
        trg_features = []
        model.eval()
        with torch.no_grad():
            for step, epoch, batch_t in trg_loader.iterator(
                    batch_size=meta_conf.batch_size,  # apply the batch-wise or sample-wise setting.
                    shuffle=False,  # we will apply shuffle operation in preparing dataset.
                    repeat=False,
                    ref_num_data=None,
                    num_workers=meta_conf.num_workers
                    if hasattr(meta_conf, "num_workers")
                    else 2,
                    pin_memory=True,
                    drop_last=False,
            ):
                # for feature visualization
                feat_t = model.forward_features(batch_t._x)
                yhat_t = model.forward_head(feat_t)

                eng_t = energy(yhat_t)

                trg_logits.append(yhat_t)
                trg_gts.append(batch_t._y)
                trg_energies.append(eng_t)
                trg_features.append(feat_t)
        model.train()
        trg_logits = torch.concat(trg_logits, dim=0)
        trg_gts = torch.concat(trg_gts, dim=0)
        trg_energies = torch.concat(trg_energies, dim=0)
        trg_features = torch.concat(trg_features, dim=0)

    if num_past_batches == 0:
        assert not os.path.isfile(full_file_name)
        np.savez(full_file_name,
                 logits_s=np.array(src_logits.cpu()),
                 energys_s=np.array(src_energies.cpu()),
                 gts_s=np.array(src_gts.cpu()),
                 feats_s=np.array(src_features.cpu()),

                 logits_t=np.array(trg_logits.cpu()),
                 energys_t=np.array(trg_energies.cpu()),
                 gts_t=np.array(trg_gts.cpu()),
                 feats_t=np.array(trg_features.cpu()),
                 )

    elif num_past_batches % 10 == 0:
        assert os.path.isfile(full_file_name)
        saved_array = np.load(full_file_name)
        logits_s_save = np.concatenate((saved_array['logits_s'], np.array(src_logits.cpu())), axis=0)
        energys_s_save = np.concatenate((saved_array['energys_s'], np.array(src_energies.cpu())), axis=0)
        gts_s_save = np.concatenate((saved_array['gts_s'], np.array(src_gts.cpu())), axis=0)
        feats_s_save = np.concatenate((saved_array['feats_s'], np.array(src_features.cpu())), axis=0)

        logits_t_save = np.concatenate((saved_array['logits_t'], np.array(trg_logits.cpu())), axis=0)
        energys_t_save = np.concatenate((saved_array['energys_t'], np.array(trg_energies.cpu())), axis=0)
        gts_t_save = np.concatenate((saved_array['gts_t'], np.array(trg_gts.cpu())), axis=0)
        feats_t_save = np.concatenate((saved_array['feats_t'], np.array(trg_features.cpu())), axis=0)

        np.savez(full_file_name,
                 logits_s=logits_s_save,
                 energys_s=energys_s_save,
                 gts_s=gts_s_save,
                 feats_s=feats_s_save,

                 logits_t=logits_t_save,
                 energys_t=energys_t_save,
                 gts_t=gts_t_save,
                 feats_t=feats_t_save,
                 )
        saved_array.close()
    else:
        pass