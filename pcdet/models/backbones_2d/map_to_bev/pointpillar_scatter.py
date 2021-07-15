import torch
import torch.nn as nn
from .memory_module import MemoryUnit, MemoryUnit_Full, MemoryUnit_Dual, MemoryUnit_Full_Agg

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

class PointPillarScatter_detr(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict
        
class PointPillarScatter_Mix(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*self.k ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        assert self.nz == 1

    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), 5, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2).reshape(nv,-1) #cat. non-attentive
        
        return points_positive

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']

        
        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            batch_mask_point = point_coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            points = point_features[batch_mask_point, :]
            points_positive = self.get_score(points, pillars)
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Mix_full(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*self.k ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        assert self.nz == 1

    def get_feature_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2).reshape(nv,-1) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        np, nc_p = points_coord[:,1:3].size()
        nv, nc_v= pillars_coord[:,-2:].size()
        assert nc_p == nc_v

        score = torch.matmul(points_coord, pillars_coord.t())# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2).reshape(nv,-1) #cat. non-attentive
        
        return points_positive

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords, mask = batch_dict['pillar_features'], batch_dict['voxel_coords'], batch_dict['pillar_mask']
        point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']

        
        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            batch_mask_point = point_coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            this_point_coords = point_coords[batch_mask_point, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            points = point_features[batch_mask_point, :]
            points_positive = self.get_feature_score(points, pillars)
            
            points_positive_coord = self.get_coord_score(points, this_point_coords, this_coords)

            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            points_positive = weight[:,0].unsqueeze(1) * points_positive + weight[:,1].unsqueeze(1) * points_positive_coord

            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Mix_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*self.k ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(points_positive, self.k)
            else:
                points_positive_ = self.memory(pillars.t(), self.k)
            points_positive = points_positive_['output']
            att = points_positive_['att']
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, self.k)
            else:
                points_positive_ = self.memory(pillars.t(), _, self.k)
            points_positive = points_positive_['output']
            att = points_positive_['att']
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Mix_Full_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*self.k ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive
        
    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)

            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            nv, nc_v= pillars_coord[:,-2:].size()
            d, nv = pillars.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, pillars_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = pillars.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    # def pairwise_dist(self,x, y):
    #     xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    #     rx = (xx.diag().unsqueeze(0).expand_as(xx))
    #     ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #     P = (rx.t() + ry - 2*zz)
    #     return P

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # k nearest voxel features (coordinate)
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, _, this_coords)
                points_positive_ = self.memory(pillars.t(), pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            points_positive_f = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive_f)))
            points_positive_c = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive_c)))
            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            pillars_aug = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c

            # pillars_aug = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars_aug)))
            
            pillars = torch.cat((pillars, pillars_aug.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Mix_Dual_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*self.k ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Dual(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive
    
    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)

            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #pillars_coord == points_coord when eval
            #points == pillars
            nv, nc= pillars_coord[:,-2:].size()
            np, nc= points_coord[:,-2:].size()
            assert nc == nc
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices
    
    # def pairwise_dist(self,x, y):
    #     xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    #     rx = (xx.diag().unsqueeze(0).expand_as(xx))
    #     ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #     P = (rx.t() + ry - 2*zz)
    #     return P

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # k nearest voxel features (coordinate)
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            points_positive_f = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive_f)))
            points_positive_c = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive_c)))
            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            pillars_aug = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c

            # pillars_aug = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars_aug)))
            
            pillars = torch.cat((pillars, pillars_aug.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory_1(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        #self.adapt_layer = nn.Linear(self.num_pt_features ,self.num_bev_features//2, bias=False)
        #self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            
            #points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory_1_kcdiff(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        coordinate sampling k different k^2 - 1 version
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.kc = self.model_cfg.NUM_KC**2
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        #self.adapt_layer = nn.Linear(self.num_pt_features ,self.num_bev_features//2, bias=False)
        #self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg_diffk(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            sc, indices = torch.topk(score.detach(), self.kc, largest=False)
            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.kc, largest=False)
            # exclude own feature, kc - 1 neighborhoods only, first nearest is itself

            points_positive = points.t()[indices[:,1:].detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k, self.kc)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k, self.kc-1)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            
            #points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory_2(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*2 ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        #self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        #self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = self.pdist(pillars_coord, points_coord)# nv X np
            # score = torch.nn.functional.softmax(score, dim=0)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points[indices.detach()] #cat. non-attentive
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            #weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            #points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            points_positive = torch.cat((points_positive_f, points_positive_c), dim=1)
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict

class PointPillarScatter_Agg_Full_Memory_3(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
        self.num_pt_features = self.model_cfg.NUM_PT_FEATURES
        self.k = self.model_cfg.NUM_K
        self.mem_size = self.model_cfg.NUM_M
        self.shrink_thres = self.model_cfg.SHRINK_TH
        self.nx, self.ny, self.nz = grid_size
        self.adapt_layer = nn.Linear(self.num_pt_features*2 ,self.num_bev_features//2, bias=False)
        self.norm_layer = nn.BatchNorm1d(self.num_bev_features//2, eps=1e-3, momentum=0.01)
        #self.weight_layer = nn.Linear(self.num_bev_features//2, 2, bias=False)
        #self.norm_layer2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01)
        self.memory = MemoryUnit_Full_Agg_3(self.mem_size, self.num_pt_features, self.shrink_thres)
        
        assert self.nz == 1


    def get_score(self, points, pillars):
        np, d = points.size()
        d, nv = pillars.size()
        
        score = torch.matmul(points, pillars)# np X nv
        
        score = torch.nn.functional.softmax(score, dim=0)

        _, indices = torch.topk(score.detach(), self.k, dim=0)

        points_positive = points[indices.detach()].permute(1,0,2) #cat. non-attentive
        
        return points_positive

    def get_coord_score(self, points, pillars, points_coord, pillars_coord):
        if self.training:
            np, nc_p = points_coord[:,1:3].size()
            nv, nc_v= pillars_coord[:,-2:].size()
            assert nc_p == nc_v
            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, pillars_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            pillars_positive = pillars.t()[indices.detach()] #cat. non-attentive #NVxKxc

            # score = torch.matmul(points_coord, pillars_coord.t())# np X nv
            score = torch.matmul(pillars_positive, points.t())# NVxKxNP
        
            score = torch.nn.functional.softmax(score, dim=2)

            _, indices = torch.topk(score.detach(), 1, dim=2)

            points_positive = points[indices.squeeze().detach()]#NVxKxC 
            
            return points_positive, indices
        else:
            #points == pillar features (no point features in eval)
            #pillars_coord == points_coord 
            nv, nc_v= pillars_coord[:,-2:].size()
            np, nc_p = points_coord[:,-2:].size()
            assert nc_p == nc_v
            d, nv = points.size()

            # score = torch.cdist(pillars_coord, pillars_coord, p=1) # nv X nv
            score = self.pdist(pillars_coord, points_coord)

            _, indices = torch.topk(score.detach(), self.k, largest=False)
            
            points_positive = points.t()[indices.detach()] #cat. non-attentive

            return points_positive, indices

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        """Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                    norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    def forward(self, batch_dict, **kwargs):
        if self.training:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        pillar_indices = [] # added for positional encoding
        att_weight = []
        batch_size = coords[:, 0].max().int().item() + 1
        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            pillar_indice = torch.zeros(
                self.num_coord_points,
                self.nz *self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            # Only include non-empty pillars
            if self.training:
                batch_mask = coords[:, 0] == batch_idx
                batch_mask_point = point_coords[:, 0] == batch_idx
                this_point_coords = point_coords[batch_mask_point, :]
            else:
                batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            if self.training:
                points = point_features[batch_mask_point, :]
                pillars_positive_coord, indices_coord = self.get_coord_score(points, pillars, this_point_coords, this_coords)
                points_positive = self.get_score(points, pillars)
                points_positive = points_positive.contiguous()
                points_positive_ = self.memory(pillars.t(), points_positive, pillars_positive_coord, self.k)
            else:
                pillars_positive_coord, indices_coord = self.get_coord_score(pillars, pillars, this_coords, this_coords)
                points_positive_ = self.memory(pillars.t(), this_coords, pillars_positive_coord, self.k)
            points_positive_f = points_positive_['output_f']
            att_f = points_positive_['att_f']
            points_positive_c = points_positive_['output_c']
            att_c = points_positive_['att_c']

            #weight
            #weight = torch.nn.functional.softmax(self.norm_layer2(self.weight_layer(pillars.t())), dim=-1)
            #points_positive = weight[:,0].unsqueeze(1) * points_positive_f + weight[:,1].unsqueeze(1) * points_positive_c
            points_positive = torch.cat((points_positive_f, points_positive_c), dim=1)
            points_positive = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(points_positive)))
            pillars = torch.cat((pillars, points_positive.t()), dim=0)
            #pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
            # import pdb; pdb.set_trace()
            #scatter
            spatial_feature[:, indices] = pillars
            pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
            pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
            pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
            batch_spatial_features.append(spatial_feature)
            pillar_indices.append(pillar_indice)
            att_weight.append(att_f)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        pillar_indices = torch.stack(pillar_indices, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_indices'] = pillar_indices
        return batch_dict
        
# class PointPillarScatter_Mix(nn.Module):
#     def __init__(self,
#                  model_cfg, grid_size, **kwargs):
#         """
#         Point Pillar's Scatter.
#         Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
#         second.pytorch.voxelnet.SparseMiddleExtractor.
#         :param output_shape: ([int]: 4). Required output shape of features.
#         :param num_input_features: <int>. Number of input features.
#         """

#         super().__init__()
#         self.model_cfg = model_cfg
#         self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
#         self.num_coord_points = self.model_cfg.NUM_COORD_POINTS
#         self.nx, self.ny, self.nz = grid_size
#         self.adapt_layer = nn.Linear(256,self.num_bev_features, bias=False)
#         self.norm_layer = nn.BatchNorm1d(self.num_bev_features, eps=1e-3, momentum=0.01)
#         assert self.nz == 1

#     def get_score(self, points, pillars):
#         np, d = points.size()
#         d, nv = pillars.size()
        
#         score = torch.matmul(points, pillars)# np X nv
        
#         score = torch.nn.functional.softmax(score, dim=0)

#         _, indices = torch.topk(score.detach(), 1, dim=0)
        
#         return indices.detach()

#     def forward(self, batch_dict, **kwargs):
#         pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
#         point_features, point_coords = batch_dict['point_features'], batch_dict['point_coords']

        
#         batch_spatial_features = []
#         pillar_indices = [] # added for positional encoding
#         batch_size = coords[:, 0].max().int().item() + 1
#         for batch_idx in range(batch_size):
#             spatial_feature = torch.zeros(
#                 self.num_bev_features,
#                 self.nz * self.nx * self.ny,
#                 dtype=pillar_features.dtype,
#                 device=pillar_features.device)
#             pillar_indice = torch.zeros(
#                 self.num_coord_points,
#                 self.nz *self.nx * self.ny,
#                 dtype=pillar_features.dtype,
#                 device=pillar_features.device
#             )
#             # Only include non-empty pillars
#             batch_mask = coords[:, 0] == batch_idx
#             batch_mask_point = point_coords[:, 0] == batch_idx
#             this_coords = coords[batch_mask, :]
#             indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
#             indices = indices.type(torch.long)
#             pillars = pillar_features[batch_mask, :]
#             pillars = pillars.t()
#             points = point_features[batch_mask_point, :]
#             pos_points = self.get_score(points, pillars)
#             points_positive = points[pos_points.detach()].squeeze(0).t()
#             pillars = torch.cat((pillars, points_positive), dim=0)
#             pillars = torch.nn.functional.relu(self.norm_layer(self.adapt_layer(pillars.t())))
#             # import pdb; pdb.set_trace()
#             #scatter
#             spatial_feature[:, indices] = pillars.t()
#             pillar_indice[0,indices] = this_coords[:,2].type(torch.float)
#             pillar_indice[1,indices] = this_coords[:,3].type(torch.float)
#             pillar_indice[2,indices] = this_coords[:,1].type(torch.float)
#             batch_spatial_features.append(spatial_feature)
#             pillar_indices.append(pillar_indice)

#         batch_spatial_features = torch.stack(batch_spatial_features, 0)
#         pillar_indices = torch.stack(pillar_indices, 0)
#         batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
#         pillar_indices = pillar_indices.view(batch_size,self.num_coord_points * self.nz, self.ny, self.nx)
#         batch_dict['spatial_features'] = batch_spatial_features
#         batch_dict['spatial_indices'] = pillar_indices
#         return batch_dict
       