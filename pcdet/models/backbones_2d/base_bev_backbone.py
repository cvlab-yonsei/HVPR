import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        # import pdb; pdb.set_trace()
        if len(self.deblocks) > 0:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        else:
            x = ups[-1]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict



# class BaseBEVBackbone_Scale(nn.Module):
#     def __init__(self, model_cfg, input_channels):
#         super().__init__()
#         self.model_cfg = model_cfg

#         if self.model_cfg.get('LAYER_NUMS', None) is not None:
#             assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
#             layer_nums = self.model_cfg.LAYER_NUMS
#             layer_strides = self.model_cfg.LAYER_STRIDES
#             num_filters = self.model_cfg.NUM_FILTERS
#         else:
#             layer_nums = layer_strides = num_filters = []

#         if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
#             assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
#             num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
#             upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
#         else:
#             upsample_strides = num_upsample_filters = []

#         if self.model_cfg.get('NUM_SCALE_FILTERS', None) is not None:
#             assert len(self.model_cfg.NUM_SCALE_FILTERS) == len(self.model_cfg.LAYER_STRIDES)
#             num_scale_filters = self.model_cfg.NUM_SCALE_FILTERS
#         else:
#             num_scale_filters = []

#         num_levels = len(layer_nums)
#         c_in_list = [input_channels, *num_filters[:-1]]
#         c_in_scale_list = [input_channels//4, *num_scale_filters[:-1]]
#         self.downblocks = nn.ModuleList()
#         # self.convblocks = nn.ModuleList()
#         self.convblocks = nn.ModuleList()
#         self.deblocks = nn.ModuleList()
#         self.scale_layers = nn.ModuleList()
#         for idx in range(num_levels):
#             self.downblocks.append(nn.Sequential(
#                 nn.ZeroPad2d(1),
#                 nn.Conv2d(
#                     c_in_list[idx], num_filters[idx], kernel_size=3,
#                     stride=layer_strides[idx], padding=0, bias=False
#                 ),
#                 nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                 nn.ReLU()
#             ))
#             cur_conv_layers=nn.ModuleList()
#             for k in range(layer_nums[idx]):
#                 cur_conv_layers.append(nn.Sequential(
#                     nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
#                     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                     nn.ReLU()
#                 ))
#             self.convblocks.append(cur_conv_layers)
#             if len(upsample_strides) > 0:
#                 stride = upsample_strides[idx]
#                 if stride >= 1:
#                     self.deblocks.append(nn.Sequential(
#                         nn.ConvTranspose2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             upsample_strides[idx],
#                             stride=upsample_strides[idx], bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))
#                 else:
#                     stride = np.round(1 / stride).astype(np.int)
#                     self.deblocks.append(nn.Sequential(
#                         nn.Conv2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             stride,
#                             stride=stride, bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))
#             if len(num_scale_filters) > 0:
#                 self.scale_layers.append(nn.Sequential(
#                     nn.ZeroPad2d(1),
#                     nn.Conv2d(
#                     c_in_scale_list[idx], num_scale_filters[idx], kernel_size=3,
#                     stride=layer_strides[idx], padding=0, bias=False
#                     ),
#                     nn.BatchNorm2d(num_scale_filters[idx], eps=1e-3, momentum=0.01),
#                     nn.ReLU()
#                 ))

#         c_in = sum(num_upsample_filters)
#         if len(upsample_strides) > num_levels:
#             self.deblocks.append(nn.Sequential(
#                 nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
#                 nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
#                 nn.ReLU(),
#             ))

#         self.num_bev_features = c_in
#         self.attention = SpatialAttention()
#     def forward(self, data_dict):
#         """
#         Args:
#             data_dict:
#                 spatial_features
#         Returns:
#         """
#         spatial_features = data_dict['spatial_features']
#         spatial_scale_features = data_dict['spatial_scale_features']
#         ups = []
#         downs=[]
#         scale_downs=[]
#         ret_dict = {}
#         x = spatial_features
#         y = spatial_scale_features
#         #for i in range(len(self.downblocks)):
#         #    x_orig = self.downblocks[i](x)
#         #    y = self.scale_layers[i](y)
#         #    for j in range(len(self.convblocks[i])):
#         #        x = self.convblocks[i][j](x)
#         #        x = self.attention(x,y)
#         #    stride = int(spatial_features.shape[2] / x.shape[2])
#         #    ret_dict['spatial_features_%dx' % stride] = x
#         #    if len(self.deblocks) > 0:
#         #        ups.append(self.deblocks[i](x))
#         #    else:
#         #        ups.append(x)
#         for i in range(len(self.downblocks)):
#             x = self.downblocks[i](x)
#             y = self.scale_layers[i](y)
#             downs.append(x)
#             scale_downs.append(y)
#         for i in range(len(self.downblocks)):
#             x = downs[i]
#             y = scale_downs[i]
#             for j in range(len(self.convblocks[i])):
#                 x = self.convblocks[i][j](x)
#                 x = self.attention(x,y)
#             stride = int(spatial_features.shape[2] / x.shape[2])
#             ret_dict['spatial_features_%dx' % stride] = x
#             if len(self.deblocks) > 0:
#                 ups.append(self.deblocks[i](x))
#             else:
#                 ups.append(x)
#         # import pdb; pdb.set_trace()
#         if len(self.deblocks) > 0:
#             x = torch.cat(ups, dim=1)
#         elif len(ups) == 1:
#             x = ups[0]
#         else:
#             x = ups[-1]

#         if len(self.deblocks) > len(self.downblocks):
#             x = self.deblocks[-1](x)

#         data_dict['spatial_features_2d'] = x

#         return data_dict
# new version 2 (channel reduction - spatial attention(full) - channel up)

class BaseBEVBackbone_Scale(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
            sfm_layer_nums = self.model_cfg.SFM_LAYER_NUMS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        if self.model_cfg.get('NUM_SCALE_FILTERS', None) is not None:
            assert len(self.model_cfg.NUM_SCALE_FILTERS) == len(self.model_cfg.LAYER_STRIDES)
            num_scale_filters = self.model_cfg.NUM_SCALE_FILTERS
        else:
            num_scale_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        c_in_scale_list = [input_channels//4, *num_scale_filters[:-1]]
        # self.downblocks = nn.ModuleList()
        # self.convblocks = nn.ModuleList()
        self.sfm_layer_nums = sfm_layer_nums
        self.sfmblocks_down = nn.ModuleList()
        self.sfmblocks_up = nn.ModuleList()
        self.scale_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            
            self.sfmblocks_down.append(nn.Sequential(
                nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))
            # self.sfmblocks_up.append(nn.Sequential(
            #     nn.Conv2d(num_scale_filters[idx], num_filters[idx], kernel_size=1, padding=0, bias=False),
            #     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
            #     nn.ReLU()
            # ))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
            if len(num_scale_filters) > 0:
                self.scale_layers.append(nn.Sequential(
                    nn.ZeroPad2d(1),
                    nn.Conv2d(
                    c_in_scale_list[idx], num_scale_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                    ),
                    nn.BatchNorm2d(num_scale_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.attention = SpatialAttention()
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        if self.training:
            spatial_features = data_dict['spatial_features']
            #####
            spatial_features_point = data_dict['spatial_features_point']
            ######
            spatial_scale_features = data_dict['spatial_scale_features']
            ups = []
            ups_point =[]
            ret_dict = {}
            # multi_scale = []
            # multi_scale_point = []
            # scale = []
            x = spatial_features
            x_point = spatial_features_point
            y = spatial_scale_features
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                x_point = self.blocks[i](x_point)
                y = self.scale_layers[i](y)
                # multi_scale.append(x)
                # multi_scale_point.append(x_point)
                # scale.append(y)
                x_att = x
                x_att_pt = x_point

                for j in range(self.sfm_layer_nums[i]):
                    x_att_ = self.sfmblocks_down[i](x_att)
                    x_att_ = self.attention(x_att_,y)
                    # x_att = self.sfmblocks_up[i](x_att)
                    x_att = x_att_ + x_att
                    x_att_pt_ = self.sfmblocks_down[i](x_att_pt)
                    x_att_pt_ = self.attention(x_att_pt_,y)
                    # x_att_pt = self.sfmblocks_up[i](x_att_pt)
                    x_att_pt = x_att_pt_ + x_att_pt
                stride = int(spatial_features.shape[2] / x.shape[2])
                ret_dict['spatial_features_%dx' % stride] = x_att
                if len(self.deblocks) > 0:
                    ups.append(self.deblocks[i](x_att))
                    ups_point.append(self.deblocks[i](x_att_pt))
                else:
                    ups.append(x_att)
                    ups_point.append(x_att_pt)
    
            # import pdb; pdb.set_trace()
            if len(self.deblocks) > 0:
                x = torch.cat(ups, dim=1)
                x_point = torch.cat(ups_point, dim=1)
            elif len(ups) == 1:
                x = ups[0]
            else:
                x = ups[-1]

            if len(self.deblocks) > len(self.blocks):
                x = self.deblocks[-1](x)

            data_dict['spatial_features_2d'] = x
            data_dict['spatial_features_point_2d'] = x_point

            return data_dict
        else:
            spatial_features = data_dict['spatial_features']
            spatial_scale_features = data_dict['spatial_scale_features']
            ups = []
            ret_dict = {}
            x = spatial_features
            y = spatial_scale_features
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                y = self.scale_layers[i](y)
                x_att = x
                for j in range(self.sfm_layer_nums[i]):
                    x_att_ = self.sfmblocks_down[i](x_att)
                    x_att_ = self.attention(x_att_,y)
                    # x_att = self.sfmblocks_up[i](x_att)
                    x_att = x_att_ + x_att
                stride = int(spatial_features.shape[2] / x.shape[2])
                ret_dict['spatial_features_%dx' % stride] = x_att
                if len(self.deblocks) > 0:
                    ups.append(self.deblocks[i](x_att))
                else:
                    ups.append(x_att)
    
            # import pdb; pdb.set_trace()
            if len(self.deblocks) > 0:
                x = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x = ups[0]
            else:
                x = ups[-1]

            if len(self.deblocks) > len(self.blocks):
                x = self.deblocks[-1](x)

            data_dict['spatial_features_2d'] = x

            return data_dict