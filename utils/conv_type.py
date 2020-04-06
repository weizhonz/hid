import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod

    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# # Not learning weights, finding subnet
# class SubnetConv(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # self.weight.data = torch.mul(self.weight.data, 0.01)
#         self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
#         torch.nn.init.zeros_(self.mask)
#
#
#     def set_prune_rate(self, prune_rate):
#         self.prune_rate = prune_rate #prune_rate is the weights remained
#         with torch.no_grad():
#             m = torch.Tensor(self.weight.size()).uniform_() < self.prune_rate
#             self.mask.masked_fill_(m, 1)
#
#     @property
#     def clamped_scores(self):
#         return self.scores.abs()
#
#     def forward(self, x):
#         w = self.weight * self.mask
#         x = F.conv2d(
#             x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
#         )
#         return x


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Continuous Sample 
"""

class ContinuousSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        # if self.training:
        #     print("in training")
        #     eps = 1e-20
        #     temp = parser_args.T
        #     uniform0 = torch.rand_like(self.scores)
        #     uniform1 = torch.rand_like(self.scores)
        #     noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
        #     # g0 = torch.as_tensor(np.random.gumbel(size=self.scores.size()), dtype=torch.float, device=torch.device('cuda'))
        #     # g1 = torch.as_tensor(np.random.gumbel(size=self.scores.size()), dtype=torch.float, device=torch.device('cuda'))
        #     subnet = torch.sigmoid((self.scores + noise)/temp)
        # else:
        #     print("in evaluate")
        #     subnet = GetSubnet.apply(self.scores, 0.46)


        print("in training")
        eps = 1e-20
        temp = parser_args.T
        uniform0 = torch.rand_like(self.scores)
        uniform1 = torch.rand_like(self.scores)
        noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
        subnet1 = torch.sigmoid((self.scores + noise)/temp)
        print("percent < 0.01: ", (self.clamped_scores < parser_args.D).float().mean().item())
        print("percent > 0.99: ", (self.clamped_scores > (1 - parser_args.D)).float().mean().item())
        print("in evaluate")
        subnet2 = (torch.rand_like(self.scores) < self.clamped_scores).float()
        print("subnet2 left: ", subnet2.mean().item())
        print("in evaluate_sort")
        pr = 0.0
        for _ in range(10):
            pr += (
                (torch.rand_like(self.clamped_scores) < self.clamped_scores)
                    .float()
                    .mean()
                    .item()
            )
        pr /= 10.0
        print("prune rate: ", pr)
        pr2 = self.clamped_scores.mean().item()
        print("prune rate2: ", pr2)
        subnet3 = GetSubnet.apply(self.scores, pr2)
        print("subnet3 left: ", subnet3.mean().item())
        # print(self.clamped_scores)
        print("difference rate: ", (subnet2 != subnet3).float().mean().item())
        subnet = None
        if self.training:
            subnet = subnet1
        else:
            print("use mask3")
            subnet = subnet3

        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

"""
Score function estimator
"""

class SFESubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, x):
        eps = 1e-20
        uniform0 = torch.rand_like(self.scores)
        uniform1 = torch.rand_like(self.scores)
        noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)

        subnet = (self.scores + noise > 0).float()
        # print("subnet", subnet)
        self.scores.grad = subnet - torch.sigmoid(self.scores)
        # print("sigmoid(scores)", torch.sigmoid(self.scores))
        # print("scores.grad", self.scores.grad)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

