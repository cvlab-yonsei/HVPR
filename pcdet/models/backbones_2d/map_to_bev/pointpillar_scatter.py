import torch
import torch.nn as nn
from .memory_module import MemoryUnit_Agg

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_Agg_Memory_1_scale(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        Masked version
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.num_scale_features = self.model_cfg.NUM_SCALE_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        # self.conf_layer = nn.Linear(self.num_pt_features ,1, bias=False)
        # self.norm_layer = nn.BatchNorm1d(1, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1

    def get_score(self, points, pillars):
        #mask added nv*nvp*1
        np, d = points.size()
        d, nv = pillars.size()
        #pillars and points
        score = torch.matmul(points, pillars)# np X nv
        score = torch.nn.functional.softmax(score, dim=0) 
       
        _, indices = torch.topk(score.detach(), self.k, dim=0)
        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        agg_weight = torch.matmul(pillars.t().unsqueeze(1), points_positive.permute(0,2,1)).squeeze() #(NV*k)
     
        agg_weight = torch.nn.functional.softmax(agg_weight, dim=1) #NVxK
        output = agg_weight.detach().unsqueeze(2) * points_positive # NVxKxC
        output = output.sum(dim=1) #NVxC

        return {'output': output, 'att': score}
        

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features,pillar_scale_features, mask, coords = (batch_dict['pillar_features'], batch_dict['pillar_scale_features'],
                                                                                        batch_dict['pillar_mask'], batch_dict['voxel_coords'])
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']

            batch_spatial_features = []
            batch_spatial_scale_features = []
            batch_spatial_features_point = []
            batch_positive_point = []
            batch_positive_memory = []

            att_weight = []
            batch_size = coords[:, 0].max().int().item() + 1

            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
                spatial_scale_feature = torch.zeros(
                    self.num_scale_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_scale_features.dtype,
                    device=pillar_scale_features.device)
             
                spatial_feature_point = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
            # Only include non-empty pillars
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
            
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                pillars_scale = pillar_scale_features[batch_mask, :]
                pillars_scale = pillars_scale.t()
                pillars_mask = mask[batch_mask, :]
                
                points = point_features[batch_mask_point, :]
                points_positive_ = self.get_score(points, pillars)
                points_memory = self.memory(pillars.t(), self.k)
                points_positive = points_positive_['output']
                att = points_memory['att']
                points_positive_mem = points_memory['output']
                pillars_point = torch.cat((pillars, points_positive.t()), dim=0)
                spatial_feature_point[:, indices] = pillars_point

                pillars = torch.cat((pillars.detach(), points_positive_mem.t()), dim=0)
                #scatter
                spatial_feature[:, indices] = pillars
                spatial_scale_feature[:, indices] = pillars_scale
         
                batch_spatial_features_point.append(spatial_feature_point)
                batch_positive_point.append(points_positive)
                batch_positive_memory.append(points_positive_mem)
                batch_spatial_features.append(spatial_feature)
                batch_spatial_scale_features.append(spatial_scale_feature)
                att_weight.append(att)
        
        
            batch_spatial_features_point = torch.stack(batch_spatial_features_point, 0)
            batch_positive_point = torch.cat(batch_positive_point, 0)
            batch_positive_memory = torch.cat(batch_positive_memory, 0)
            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_scale_features = torch.stack(batch_spatial_scale_features, 0)
            batch_spatial_features_point = batch_spatial_features_point.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_spatial_scale_features = batch_spatial_scale_features.view(batch_size, self.num_scale_features * self.nz, self.ny, self.nx)
        
            batch_dict['spatial_features'] = batch_spatial_features
            batch_dict['spatial_features_point'] = batch_spatial_features_point
            batch_dict['spatial_scale_features'] = batch_spatial_scale_features
            batch_dict['point_positive_features'] = batch_positive_point
            batch_dict['memory_positive_features'] = batch_positive_memory
            batch_dict['memory_items'] = self.memory.weight
        
        else:
            pillar_features,pillar_scale_features, mask, coords = (batch_dict['pillar_features'], batch_dict['pillar_scale_features'],
                                                                                        batch_dict['pillar_mask'], batch_dict['voxel_coords'])

            batch_spatial_features = []
            batch_spatial_scale_features = []
            att_weight = []
            batch_size = coords[:, 0].max().int().item() + 1

            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
                spatial_scale_feature = torch.zeros(
                    self.num_scale_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_scale_features.dtype,
                    device=pillar_scale_features.device)
     
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                pillars_scale = pillar_scale_features[batch_mask, :]
                pillars_scale = pillars_scale.t()
                pillars_mask = mask[batch_mask, :]
                
                points_memory = self.memory(pillars.t(), self.k)
                att = points_memory['att']
                points_positive_mem = points_memory['output']

                pillars = torch.cat((pillars.detach(), points_positive_mem.t()), dim=0)
           
                #scatter
                spatial_feature[:, indices] = pillars
                spatial_scale_feature[:, indices] = pillars_scale
         
                batch_spatial_features.append(spatial_feature)
                batch_spatial_scale_features.append(spatial_scale_feature)
                att_weight.append(att)
        
       
            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_scale_features = torch.stack(batch_spatial_scale_features, 0)
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_spatial_scale_features = batch_spatial_scale_features.view(batch_size, self.num_scale_features * self.nz, self.ny, self.nx)
            batch_dict['spatial_features'] = batch_spatial_features
            batch_dict['spatial_scale_features'] = batch_spatial_scale_features
        
        return batch_dict