import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Most from CBAM official repo .
  
"""
class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 use_norm=True,
                 activation=False
                 ):
        super().__init__()

        self.use_norm = use_norm
        self.activation = activation

        if self.use_norm:
            self.conv = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=kernel_size, stride=stride, padding=padding)

        if self.activation:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, use_norm=True, activation=False)
    def forward(self, x, w):
        att = self.compress(w)
        # feature = self.compress(x)
        # att = torch.cat((scale, feature), dim=1)
        att = self.spatial(att)
        att = F.sigmoid(att) # broadcasting
        return att * x 
        
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, use_norm=True, activation=False)
#     def forward(self, x, w):
#         scale = self.compress(w)
#         scale = self.spatial(scale)
#         scale = F.sigmoid(scale) # broadcasting
#         return scale * x + x
