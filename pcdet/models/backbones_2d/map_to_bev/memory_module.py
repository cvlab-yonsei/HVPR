from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

# Modified from official MemAE repo.

class MemoryUnit_Agg(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit_Agg, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, k):
        # input1: pillar features / input2: k positive point features for each pillar
        if self.training:
            nv, d = input1.size()
            nv,_, d = input2.size()
            points = input2.reshape(-1,d)
            mem_trans = self.weight.permute(1, 0)  # Mem^T, CxM
            # get nearest memory items for each k points features
            att_weight = F.linear(points, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
            att_weight = F.softmax(att_weight, dim=1)  # TxM
            
            # ReLU based shrinkage, hard shrinkage for positive value
            if(self.shrink_thres>0):
                att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
                # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
                # normalize???
                att_weight = F.normalize(att_weight, p=1, dim=1)
                # att_weight = F.softmax(att_weight, dim=1)
                # att_weight = self.hard_sparse_shrink_opt(att_weight)
            
            memory_positive = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
            memory_positive = memory_positive.reshape(nv,k,d)

            #aggregate 
            pillars = input1.unsqueeze(1).expand(nv,k,d)
            agg_weight = (memory_positive * pillars).sum(dim=2) #(NVxKxC) * (NVxKxC) =  (NVxKxC) /sum NVxK
            agg_weight = F.softmax(agg_weight, dim=1) #NVxK
            output = agg_weight.detach().unsqueeze(2).expand(nv,k,d) * memory_positive # NVxKxC
            output = output.sum(dim=1) #NVxC
        
            return {'output': output, 'att': att_weight}  # output, att_weight
        else:
            #input2 not used in eval (no point features, any tensor)
            nv, d = input1.size()
            #pick k nearest items
            score = F.linear(input1, self.weight)# nv X M
            score = F.softmax(score, dim=1)
            _, indices = torch.topk(score.detach(), k, dim=1)
            memory_positive = self.weight[indices.detach()] #NVxKxC
            
            #aggregate 
            pillars = input1.unsqueeze(1).expand(nv,k,d)
            agg_weight = (memory_positive * pillars).sum(dim=2) #(NVxKxC) * (NVxKxC) =  (NVxKxC) /sum NVxK
            agg_weight = F.softmax(agg_weight, dim=1) #NVxK
            output = agg_weight.detach().unsqueeze(2).expand(nv,k,d) * memory_positive # NVxKxC
            output = output.sum(dim=1) #NVxC
            Mem, (TxM) x (MxC) = TxC
            
            return {'output': output, 'att': score}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output