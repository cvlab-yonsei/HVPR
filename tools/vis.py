import cv2
import numba
import numpy as np
import matplotlib.pyplot as plt
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator

@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        # coors_2d,
        bev_map,
        height_lowers,
        # density_norm_num=16,
        with_reflectivity=False,
        max_voxels=40000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3, ), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(points,
                  voxel_size,
                  coors_range,
                  with_reflectivity=False,
                  density_norm_num=16,
                  max_voxels=40000):
    """convert kitti points(N, 4) to a bev map. return [C, H, W] map.
    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor. 
            `WARNING`: bev_map[-1] is num_points map, NOT density map, 
            because calculate density map need more time in cpu rather than gpu. 
            if with_reflectivity is True, bev_map[-2] is intensity map. 
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False)
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(points, voxel_size, coors_range,
                                     coor_to_voxelidx, bev_map, height_lowers,
                                     with_reflectivity, max_voxels)
    # print(voxel_num)
    return bev_map


def point_to_vis_bev(points,
                     voxel_size=None,
                     coors_range=None,
                     max_voxels=80000):
    if voxel_size is None:
        voxel_size = [0.1, 0.1, 0.1]
    if coors_range is None:
        coors_range = [-50, -50, -3, 50, 50, 1]
    voxel_size[2] = coors_range[5] - coors_range[2]
    bev_map = points_to_bev(
        points, voxel_size, coors_range, max_voxels=max_voxels)
    height_map = (bev_map[0] * 255).astype(np.uint8)
    return cv2.cvtColor(height_map, cv2.COLOR_GRAY2RGB)


def cv2_draw_lines(img, lines, colors, thickness, line_type=cv2.LINE_8):
    lines = lines.astype(np.int32)
    for line, color in zip(lines, colors):
        color = list(int(c) for c in color)
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img


def cv2_draw_text(img, locs, labels, colors, thickness, line_type=cv2.LINE_8):
    locs = locs.astype(np.int32)
    font_line_type = cv2.LINE_8
    font = cv2.FONT_ITALIC
    font = cv2.FONT_HERSHEY_DUPLEX
    font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    for loc, label, color in zip(locs, labels, colors):
        color = list(int(c) for c in color)
        cv2.putText(img, label, tuple(loc), font, 0.7, color, thickness,
                    font_line_type, False)
    return img

def corner_to_standup_nd(boxes_corner):

    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)
    
def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

def draw_box_in_bev(img,
                    coors_range,
                    boxes,
                    color,
                    thickness=1,
                    labels=None,
                    label_color=None):
    """
    Args:
        boxes: center format.
    """
    coors_range = np.array(coors_range)
    bev_corners = center_to_corner_box2d(
        boxes[:, [0, 1]], boxes[:, [3, 4]], boxes[:, 6])
    bev_corners -= coors_range[:2]
    bev_corners *= np.array(
        img.shape[:2])[::-1] / (coors_range[3:5] - coors_range[:2])
    standup = corner_to_standup_nd(bev_corners)
    text_center = standup[:, 2:]
    text_center[:, 1] -= (standup[:, 3] - standup[:, 1]) / 2

    bev_lines = np.concatenate(
        [bev_corners[:, [0, 2, 3]], bev_corners[:, [1, 3, 0]]], axis=2)
    bev_lines = bev_lines.reshape(-1, 4)
    colors = np.tile(np.array(color).reshape(1, 3), [bev_lines.shape[0], 1])
    colors = colors.astype(np.int32)
    img = cv2_draw_lines(img, bev_lines, colors, thickness)
    if boxes.shape[1] == 9:
        # draw velocity arrows
        for box in boxes:
            velo = box[-2:]
            # velo = np.array([-np.sin(box[6]), -np.cos(box[6])])
            velo_unified = velo
            if np.isnan(velo_unified[0]):
                continue
            velo_unified = velo_unified * np.array(
                img.shape[:2])[::-1] / (coors_range[3:5] - coors_range[:2])
            center = box[:2] - coors_range[:2]
            center = center * np.array(
                img.shape[:2])[::-1] / (coors_range[3:5] - coors_range[:2])
            center = tuple(map(lambda x: int(x), center))
            center2 = tuple(map(lambda x: int(x), center + velo_unified))
            cv2.arrowedLine(img, center, center2, color, thickness, tipLength=0.3)
    if labels is not None:
        if label_color is None:
            label_color = colors
        else:
            label_color = np.tile(
                np.array(label_color).reshape(1, 3), [bev_lines.shape[0], 1])
            label_color = label_color.astype(np.int32)

        img = cv2_draw_text(img, text_center, labels, label_color,
                            thickness)
    return img


def kitti_vis(points, boxes=None, labels=None):
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [0, -30, -3, 64, 30, 1]
    bev_map = point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    if boxes is not None:
        bev_map = draw_box_in_bev(bev_map, vis_point_range, boxes, [0, 255, 0], 2, labels)

    return bev_map

def nuscene_vis(points, boxes=None, labels=None):
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -50, -5, 50, 50, 3]
    bev_map = point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    if boxes is not None:
        bev_map = draw_box_in_bev(bev_map, vis_point_range, boxes, [0, 255, 0], 2, labels)

    return bev_map

def nuscene_vis_(points, boxes=None, labels=None):
#     vis_voxel_size = [0.1, 0.1, 0.1]
    vis_voxel_size = [0.16, 0.16, 0.1]
#     vis_voxel_size = [0.08, 0.08, 0.1]
    vis_point_range = [0, -39.68, -3, 69.12, 39.68, 1]
    bev_map = point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    if boxes is not None:
        bev_map = draw_box_in_bev(bev_map, vis_point_range, boxes, [0, 255, 0], 2, labels)

    return bev_map

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def break_up_pc(pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=None, training=False
    )
    c = 50
    dataloader_iter = iter(test_loader)
    batch = next(dataloader_iter)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    model_func = model_fn_decorator()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            # data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(batch)
            data_dict,_, batch_dict = model.forward(batch)
            # import pdb; pdb.set_trace()

            boxes = data_dict[0]['pred_boxes'].cpu().numpy()
            points = batch_dict['points']
            gt_boxes = batch['gt_boxes'][0][:,:].cpu().numpy()
            batch_idx, xyz, features = break_up_pc(points)
            points = torch.cat((xyz,features), dim=1).cpu().numpy()
            bev_map_gt = nuscene_vis_(points, gt_boxes)
            bev_map = nuscene_vis_(points, boxes)
            plt.imsave('demo_50.png', bev_map)
            plt.imsave('demo_gt.png', bev_map_gt)
            if idx+1 == c:
                break
    print(batch['frame_id'])                
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
