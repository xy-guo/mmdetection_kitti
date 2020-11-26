import os
import pickle
import numpy as np
import torch.utils.data as torch_data
from collections import defaultdict
import datasets.kitti_common as kitti_common
import utils.box_np_ops as box_np_ops
import utils.geometry as geometry
import utils.box_coder as box_coder_utils
from datasets.target_assigner import TargetAssigner, AnchorGeneratorRange
from config import cfg
import imageio
import copy


class KittiStereoDataset(torch_data.Dataset):
    def __init__(self, root_path, class_names, split, training, logger=None):
        """
        :param root_path: KITTI data path
        :param split:
        """
        super().__init__()

        self.root_path = os.path.join(root_path, 'object')
        self.root_split_path = os.path.join(self.root_path, 'training' if split != 'test' else 'testing')
        self.class_names = class_names
        self.split = split
        self.training = training
        self.logger = logger
        self.img_w = cfg.INPUT_DATA.IMAGE_WIDTH
        self.img_h = cfg.INPUT_DATA.IMAGE_HEIGHT

        # read kitti infos, which is a list of dict with the following keys:
        # image_idx, velodyne_path, img_path, image_shape, calib matrices, annos (annotations)
        info_path = os.path.join(self.root_path, 'kitti_infos_%s.pkl' % split)
        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)

        # NOTE: we are not able to use sampler any more because we cannot edit stereo images for augmentation.
        # database sampler
        # self.db_sampler = None
        # if self.training and cfg.DB_SAMPLER.ENABLED:
        #     # read db info, which is a dict for all categories (Car, Ped, Cyclist, etc.)
        #     # For each category, for example 'Car', there is a list of dict with the following keys:
        #     # name, path, image_idx, gt_idx, box3d_lidar, num_points_in_gt, difficulty, group_id, etc/
        #     # actually each sample represents an object subsest of lidar points
        #     db_info_path = os.path.join(self.root_path, 'kitti_dbinfos_%s.pkl' % split)
        #     with open(db_info_path, 'rb') as f:
        #         db_infos = pickle.load(f)
        #     self.db_sampler = DataBaseSampler(db_infos=db_infos, sampler_cfg=cfg.DB_SAMPLER, logger=logger)

        # NOTE: voxel generator is replaced by cost volume generator, which is inside the backbone network
        # voxel generator: convert point cloud into a voxel grid
        # voxel_generator_cfg = cfg.VOXEL_GENERATOR
        # self.voxel_generator = VoxelGenerator(voxel_size=voxel_generator_cfg.VOXEL_SIZE,
        #                                       point_cloud_range=voxel_generator_cfg.POINT_CLOUD_RANGE,
        #                                       max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
        #                                       max_voxels=20000)

        # a list of configs for each class, car, cyclist, pedestrian, etc.
        # each config consists of the following keys:
        # anchor_range: array [6]
        # matched_threshold, unmatched_threshold
        # sizes, rotations: (for each position xyz, generate anchors with different sizes and rotations)
        anchor_cfg = cfg.TARGET_ASSIGNER.ANCHOR_GENERATOR
        # anchor generators for each class: `generate anchors` with different x,y,z,sizes,rotations
        anchor_generators = []
        for a_cfg in anchor_cfg:
            anchor_generator = AnchorGeneratorRange(
                anchor_ranges=a_cfg['anchor_range'],
                sizes=a_cfg['sizes'],
                rotations=a_cfg['rotations'],
                class_name=a_cfg['class_name'],
                match_threshold=a_cfg['matched_threshold'],
                unmatch_threshold=a_cfg['unmatched_threshold']
            )
            anchor_generators.append(anchor_generator)

        # box coder: compute the `regression target` for the outputs (according to the anchors)
        # usually it could be the residual values based on the anchor size and rotation
        # will be used in target_assigner
        self.box_coder = getattr(box_coder_utils, cfg.BOX_CODER)()

        # target assigner: assign anchors with corresponding gt boxes to obtain training labels and regression targets
        self.target_assigner = TargetAssigner(
            anchor_generators=anchor_generators,
            pos_fraction=cfg.TARGET_ASSIGNER.SAMPLE_POS_FRACTION,
            sample_size=cfg.TARGET_ASSIGNER.SAMPLE_SIZE,
            region_similarity_fn_name=cfg.TARGET_ASSIGNER.REGION_SIMILARITY_FN,
            box_coder=self.box_coder
        )

        # the number of anchors should be D/dd/stride * W/dw/stride * 2 (angle), 7
        # generate cached anchors
        # because anchors are fixed, generate beforehand to save computation cost
        # compute the output size factor of the 3D CNN
        out_size_factor = cfg.RPN_STAGE.RPN_HEAD.LAYER_STRIDES[0] / cfg.RPN_STAGE.RPN_HEAD.UPSAMPLE_STRIDES[0]
        # out_size_factor *= cfg.RPN_STAGE.COST_VOLUME_GENERATOR.DOWNSAMPLE_RATE_3D
        out_size_factor = int(out_size_factor)
        assert out_size_factor > 0
        # the grid size of the initial cost volume
        grid_size = np.array(cfg.RPN_STAGE.COST_VOLUME.GRID_SIZE, dtype=np.int64)
        # the feature map size after the cost volume is processed by the 3D CNN
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors_dict = self.target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"].reshape([-1, 7])
        self.anchor_cache = {
            "anchors": anchors,
            "anchors_dict": anchors_dict,
        }

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne', '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]
        points = points[points[:, 0] > 0]
        return points

    def get_lidar_reduced(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne_reduced', '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx, left_or_right='left'):
        if left_or_right == 'left':
            tag = 'image_2'
        elif left_or_right == 'right':
            tag = 'image_3'
        else:
            raise ValueError('can only be left or right')
        img_file = os.path.join(self.root_split_path, tag, '%06d.png' % idx)
        assert os.path.exists(img_file), 'img_file: %s not exist' % img_file
        img = imageio.imread(img_file)
        img = img.astype(np.float32) / 255.
        img = img * 2 - 1
        return img

    def get_depth_image(self, idx, left_or_right='left', img_shape=None):
        if left_or_right == 'left':
            tag = 'depth_2'
        elif left_or_right == 'right':
            tag = 'depth_3'
        else:
            raise ValueError('can only be left or right')
        img_file = os.path.join(self.root_split_path, tag, '%06d.png' % idx)
        if not os.path.exists(img_file):
            print('img_file: %s not exist' % img_file)
            return None
        img = imageio.imread(img_file)
        assert np.max(img) > 255
        img = img.astype(np.float32) / 256.
        img[img == 0] = np.nan
        return img

    # def get_road_plane(self, idx):
    #     plane_file = os.path.join(self.root_split_path, 'planes', '%06d.txt' % idx)
    #     with open(plane_file, 'r') as f:
    #         lines = f.readlines()
    #     lines = [float(i) for i in lines[3].split()]
    #     plane = np.asarray(lines)
    #
    #     # Ensure normal is always facing up, this is in the rectified camera coordinate
    #     if plane[1] > 0:
    #         plane = -plane
    #
    #     norm = np.linalg.norm(plane[0:3])
    #     plane = plane / norm
    #     return planes

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, index):
        info = self.kitti_infos[index]
        sample_idx = info['image_idx']
        rect = info['calib/R0_rect'].astype(np.float32)
        Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib/P2'].astype(np.float32)
        P3 = info['calib/P3'].astype(np.float32)

        points = self.get_lidar(sample_idx)
        left_img = self.get_image(sample_idx, 'left')
        right_img = self.get_image(sample_idx, 'right')
        left_depth = self.get_depth_image(sample_idx, 'left')
        right_depth = self.get_depth_image(sample_idx, 'right')
        if left_depth is not None:
            assert left_img.shape[:2] == left_depth.shape[:2]

        input_dict = {
            'left_img': left_img,
            'right_img': right_img,
            'left_depth': left_depth,
            'right_depth': right_depth,
            'points': points,
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'P3': P3,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            'sample_idx': sample_idx,
            'image_path': info['img_path']
        }

        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = kitti_common.remove_dontcare(annos)
            loc = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            # print(gt_names, len(loc))
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
            difficulty = annos["difficulty"]
            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'difficulty': difficulty,
            })

        example = self.prepare_data(input_dict=input_dict)
        example["sample_idx"] = sample_idx
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def convert_points_to_grid(self, points):
        """

        :param points [N, 3]: should be in camera coordinate (cam0 or cam2)
        :return: voxel_grid [N_h, N_w, N_d], inverse format of VOXEL_SIZE in config.py
        """
        points = box_np_ops.camera_to_pseudo_lidar(points)  # convert into D, W, H order (pseudo lidar coordinate)
        cost_volume_cfg = cfg.RPN_STAGE.COST_VOLUME
        self.voxel_size = np.array(cost_volume_cfg.VOXEL_SIZE)  # size of each voxel
        self.point_cloud_range = np.array(cost_volume_cfg.POINT_CLOUD_RANGE)
        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

        # generate voxel centers for cost volume resampling
        voxel_grid = np.zeros(self.grid_size, dtype=np.float32)  # in D, W, H order
        voxel_ids = np.round((points - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int64)
        valid = np.logical_and(np.all(voxel_ids >= 0, axis=1), np.all(voxel_ids < self.grid_size, axis=1))
        voxel_ids = voxel_ids[valid, :]
        # print(self.grid_size, voxel_ids.min(0), voxel_ids.max(0))
        np.add.at(voxel_grid, (voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]), 1)
        voxel_grid = np.transpose(voxel_grid, [2, 1, 0])
        return voxel_grid

    def convert_points_to_depth_map(self, points, K):
        assert K.shape == (3, 3)
        depth = points[:, 2:3]
        uv = points @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        # uv -= 1  # use minus 1 to get the exact same value as KITTI matlab code
        # TODO: do not fix the size, use configs or input_dict["image_shape"]
        depth_map = np.zeros([self.img_h, self.img_w], dtype=np.float32)
        depth_map_shape = np.array(list(depth_map.shape)[::-1])
        valid = np.logical_and(np.all(uv > 0, 1), np.all(uv < depth_map_shape - 1, 1))
        valid = np.logical_and(valid, depth[:, 0] > 0)
        uv = uv[valid]
        depth = depth[valid]
        u, v, depth = uv[:, 0], uv[:, 1], depth[:, 0]
        depth_map[...] = 10000.  # set to a large value
        np.minimum.at(depth_map, (np.floor(v).astype(np.uint32), np.floor(u).astype(np.uint32)), depth)
        np.minimum.at(depth_map, (np.ceil(v).astype(np.uint32), np.floor(u).astype(np.uint32)), depth)
        np.minimum.at(depth_map, (np.floor(v).astype(np.uint32), np.ceil(u).astype(np.uint32)), depth)
        np.minimum.at(depth_map, (np.ceil(v).astype(np.uint32), np.ceil(u).astype(np.uint32)), depth)
        depth_map[depth_map > 9999] = np.nan
        return depth_map

    def adjust_image_size(self, img, K=None, fill=0.):
        # adjust the image size (h, w) to (self.img_h, self.img_w)
        h, w = img.shape[:2]
        # adjust the height
        if h < self.img_h:
            pad_top = self.img_h - h
            if len(img.shape) == 3:
                img = np.pad(img, ((pad_top, 0), (0, 0), (0, 0)), 'constant', constant_values=fill)
            else:
                img = np.pad(img, ((pad_top, 0), (0, 0)), 'constant', constant_values=fill)
        elif h > self.img_h:
            cut_top = h - self.img_h
            img = img[cut_top:, :]
        # adjust the width
        if w < self.img_w:
            pad_right = self.img_w - w
            if len(img.shape) == 3:
                img = np.pad(img, ((0, 0), (0, pad_right), (0, 0)), 'constant', constant_values=fill)
            else:
                img = np.pad(img, ((0, 0), (0, pad_right)), 'constant', constant_values=fill)
        elif w > self.img_w:
            cut_right = w - self.img_w
            img = img[:, :-cut_right]

        if K is not None:
            K[1, 2] += (self.img_h - h)
        return img, K

    def reformat_transformation_matrices(self, Trv2c, rect, P2, P3):
        K, t2 = self.decompose_kitti_intrinsics(P2)
        _, t3 = self.decompose_kitti_intrinsics(P3)
        # to be consistent with official matlab code
        K[0:2, 2] -= 1
        # transformation from cam0 to cam2
        T0to2 = np.eye(4, dtype=np.float32)
        T0to2[:3, 3:4] = t2
        # transformation from cam0 to cam3
        T0to3 = np.eye(4, dtype=np.float32)
        T0to3[:3, 3:4] = t3
        # translation from cam2 to cam3
        t2to3 = np.squeeze(t3 - t2, -1)

        # create new transformation matrix
        Trv2c2 = T0to2 @ rect @ Trv2c
        Trv2c3 = T0to3 @ rect @ Trv2c
        rect = np.eye(4, dtype=np.float32)
        return Trv2c2, Trv2c3, rect, K, t2to3, t2.squeeze(-1), t3.squeeze(-1)

    def prepare_data(self, input_dict):
        # sample_idx = input_dict['sample_idx']
        velo_points = input_dict['points']  # point cloud in velo coordinate system?
        image_shape = input_dict["image_shape"]
        ori_image_shape = np.array(input_dict["image_shape"], dtype=np.int32)
        Trv2c2, Trv2c3, rect, K, t2to3, t0to2, t0to3 = self.reformat_transformation_matrices(
                input_dict['Trv2c'], input_dict['rect'],
                input_dict['P2'], input_dict['P3'])
        ori_K = K.copy()
        # round the y, z offsets to be zero
        t2to3[..., 1:] = 0.

        if 'gt_boxes' in input_dict.keys():
            gt_boxes = input_dict["gt_boxes"]
            gt_names = input_dict["gt_names"]
            difficulty = input_dict["difficulty"]
        else:
            gt_boxes = gt_names = difficulty = None

        # pad image to (self.img_h, self.img_w) and update intrinsics accordingly
        left_img, right_img = input_dict['left_img'], input_dict['right_img']
        left_depth_img, right_depth_img = input_dict['left_depth'], input_dict['right_depth']
        left_img, K = self.adjust_image_size(left_img, K=K)
        right_img, _ = self.adjust_image_size(right_img, K=None)
        if left_depth_img is not None:
            left_depth_img, _ = self.adjust_image_size(left_depth_img, K=None, fill=np.nan)
            right_depth_img, _ = self.adjust_image_size(right_depth_img, K=None, fill=np.nan)
        image_shape = np.array([self.img_h, self.img_w], dtype=np.int32)

        # convert points from velo coordinate system to image coordinate system.
        cam2_points = box_np_ops.lidar_to_camera(velo_points, rect, Trv2c2)
        depth_map = self.convert_points_to_depth_map(cam2_points, K)

        vis = False
        if vis:
            # import matplotlib.pyplot as plt
            imageio.imwrite('/home/xyguo/img.png', left_img)
            imageio.imwrite('/home/xyguo/img-depth.png', 1 / depth_map * 3)
            exit()

        noise_T = np.eye(4, dtype=np.float32)
        if self.training:
            selected = kitti_common.drop_arrays_by_name(gt_names, ["DontCare"])
            gt_boxes = gt_boxes[selected]
            gt_names = gt_names[selected]
            difficulty = difficulty[selected]

            # NOTE: the gt boxes are converted into the pseudo lidar coordinate system to make sure
            #  x and y are the first two dimensions, which makes it convenient to use existing utility functions.
            # # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
            gt_boxes = box_np_ops.box_camera_to_psuedo_lidar(gt_boxes)
            # set cared class names to be True, not cared class names to be False
            gt_boxes_mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)

            # if self.db_sampler is not None:
            #     road_planes = self.get_road_plane(sample_idx) if cfg.DB_SAMPLER.USE_ROAD_PLANE else None
            #
            #     sampled_dict = self.db_sampler.sample_all(self.root_path, gt_boxes, gt_names, road_planes,
            #                                               rect=rect, Trv2c=Trv2c, P2=P2)
            #
            #     if sampled_dict is not None:
            #         sampled_gt_names = sampled_dict["gt_names"]
            #         sampled_gt_boxes = sampled_dict["gt_boxes"]
            #         sampled_points = sampled_dict["points"]
            #         sampled_gt_masks = sampled_dict["gt_masks"]
            #
            #         gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
            #         gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
            #         gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)
            #
            #         points = self.remove_points_in_boxes(points, sampled_gt_boxes)
            #         points = np.concatenate([sampled_points, points], axis=0)

            # if not cfg.INPUT_DATA.USE_INTENSITY:
            #     points = points[:, :3]

            # data_augment_utils.noise_per_object_v3_(
            #     gt_boxes,
            #     points,
            #     gt_boxes_mask,
            #     rotation_perturb=cfg.INPUT_DATA.GT_ROT_UNIFORM_NOISE,
            #     center_noise_std=cfg.INPUT_DATA.GT_LOC_NOISE_STD,
            #     global_random_rot_range=cfg.INPUT_DATA.GLOBAL_ROT_RANGE_PER_OBJECT,
            #     num_try=100
            # )

            # should remove unrelated objects after noise per object
            gt_boxes = gt_boxes[gt_boxes_mask]
            gt_names = gt_names[gt_boxes_mask]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

            # TODO: add random flipping
            if cfg.INPUT_DATA.FLIP_LR:  # do random flip-lr
                if np.random.random() > 0.5:
                    # update left_img, right_img, depth_map
                    cam3_points = box_np_ops.lidar_to_camera(velo_points, rect, Trv2c3)
                    left_img, right_img = right_img[:, ::-1], left_img[:, ::-1]
                    if left_depth_img is not None:
                        left_depth_img, right_depth_img = right_depth_img[:, ::-1], left_depth_img[:, ::-1]
                    depth_map = self.convert_points_to_depth_map(cam3_points, K)[:, ::-1]
                    # update cx in intrinsics
                    K[0, 2] = image_shape[1] - 1 - K[0, 2]
                    # update gt_boxes, flip y because it's in pseudo lidar coordinate system
                    gt_boxes[:, 1] -= t0to3[0]
                    gt_boxes[:, 1] *= (-1)
                    gt_boxes[:, 6] *= (-1)
                    if cfg.DEBUG:
                        cam2_points = cam3_points.copy()
                        cam2_points[..., 0] *= -1
            else:
                gt_boxes[:, 1] -= t0to2[0]

            # global rotation augmentation
            low, high = cfg.INPUT_DATA.GLOBAL_ROT_UNIFORM_NOISE
            if low < high:
                noise_rad = np.random.uniform(low, high) / 180. * np.pi
                # update gt boxes positions and angles
                gt_boxes[:, :3], noise_R = box_np_ops.rotation_points_single_angle(
                        gt_boxes[:, :3], noise_rad, axis=2, return_rot=True)
                if cfg.DEBUG:
                    cam2_points = box_np_ops.pseudo_lidar_to_camera(box_np_ops.rotation_points_single_angle(
                            box_np_ops.camera_to_pseudo_lidar(cam2_points), noise_rad, axis=2))
                gt_boxes[:, 6] += noise_rad
                # update noise transformation matrix
                noise_T[:3, :3] = noise_R
            if cfg.DEBUG:
                grid = self.convert_points_to_grid(cam2_points)
            # gt_boxes = self.augmentation_global_rotation(gt_boxes)

            # gt_boxes, points = data_augment_utils.random_flip(gt_boxes, points)
            # gt_boxes, points = data_augment_utils.global_rotation(gt_boxes, points,
            #                                                       rotation=cfg.INPUT_DATA.GLOABL_ROT_UNIFORM_NOISE)
            # gt_boxes, points = data_augment_utils.global_scaling_v2(gt_boxes, points,
            #                                                         *cfg.INPUT_DATA.GLOABL_SCALING_UNIFORM_NOISE)

            pc_range = np.array(cfg.RPN_STAGE.COST_VOLUME.POINT_CLOUD_RANGE)  # self.voxel_generator.point_cloud_range
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = self.filter_gt_box_outside_range(gt_boxes, bv_range)
            gt_boxes = gt_boxes[mask]
            gt_classes = gt_classes[mask]
            gt_names = gt_names[mask]

            # limit rad to [-pi, pi]
            gt_boxes[:, 6] = box_np_ops.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)
        else:
            if gt_boxes is not None:
                # for evaluation
                selected = kitti_common.keep_arrays_by_name(gt_names, self.class_names)
                gt_boxes = gt_boxes[selected]
                gt_names = gt_names[selected]
                # NOTE: the gt boxes are converted into the pseudo lidar coordinate system to make sure
                #  x and y are the first two dimensions, which makes it convenient to use existing utility functions.
                # # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
                gt_boxes = box_np_ops.box_camera_to_psuedo_lidar(gt_boxes)

        MODE = 'TRAIN' if self.training else 'TEST'
        # if cfg.INPUT_DATA[MODE].SHUFFLE_POINTS:
        #     # shuffle is a little slow.
        #     np.random.shuffle(points)

        # # TODO: we have set max_voxels=20000. But test with 40000, work or not?
        # voxels, coordinates, num_points = \
        #     self.voxel_generator.generate(points, max_voxels=cfg.INPUT_DATA[MODE].MAX_NUMBER_OF_VOXELS)
        # voxel_centers = coordinates[:, ::-1]  # (xyz)
        # voxel_centers = \
        #     (voxel_centers + 0.5) * self.voxel_generator.voxel_size + self.voxel_generator.point_cloud_range[0:3]

        # TODO: add left-right image here and augmentation.

        example = {
            # 'voxels': voxels,  # sparse voxel bins
            # 'num_points': num_points,  # num of points for each voxel bin?
            # 'coordinates': coordinates,  # integer coordinate for each voxel bin?
            # 'num_voxels': np.array([voxels.shape[0]], dtype=np.int64),
            # 'Trv2c2': Trv2c2,
            # 'grid': grid,
            'K': K,
            'ori_K': ori_K,
            't2to3': t2to3,
            'noise_T': noise_T
            # 't0to2': t0to2
            # 'voxel_centers': voxel_centers
        }
        if cfg.DEBUG:
            example['grid'] = grid

        anchors = self.anchor_cache["anchors"]
        anchors_dict = self.anchor_cache["anchors_dict"]
        example["anchors"] = anchors

        anchors_mask = None

        if self.training:
            targets_dict = self.target_assigner.assign_v2(anchors_dict, gt_boxes, anchors_mask,
                                                          gt_classes=gt_classes, gt_names=gt_names)

            example.update({
                'labels': targets_dict['labels'],
                'reg_targets': targets_dict['bbox_targets'],
                'reg_weights': targets_dict['bbox_outside_weights'],
            })

        if gt_boxes is not None:
            example["gt_boxes"] = gt_boxes

        example['left_img'] = left_img
        example['right_img'] = right_img
        example["image_shape"] = image_shape
        example["ori_image_shape"] = ori_image_shape
        example["depth_map"] = depth_map
        example["depth_map_filtered"] = left_depth_img if left_depth_img is not None else depth_map

        return example

    @staticmethod
    def decompose_kitti_intrinsics(P):
        """

        :param P: [4, 4]
        :return: intrinsics K [3, 3], and translation t [3, 1]
        """
        assert P.shape == (4, 4)
        K = P[:3, :3]
        Kt = P[:3, 3:4]
        t = np.linalg.inv(K) @ Kt
        return K, t

    @staticmethod
    def compose_kitti_intrinsics(K, t):
        """

        :param K: intrinsics [3, 3]
        :param t: translation t [3, 1]
        :return: P: [4, 4]
        """
        P = np.eye(4, dtype=K.dtype)
        P[:3, :3] = K
        P[:3, 3:4] = K @ t
        return P

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    @staticmethod
    def filter_gt_box_outside_range(gt_boxes, limit_range):
        """remove gtbox outside training range.
        this function should be applied after other prep functions
        Args:
            gt_boxes ([type]): [description]
            limit_range ([type]): [description]
        """
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, 6])
        bounding_box = box_np_ops.minmax_to_corner_2d(
            np.asarray(limit_range)[np.newaxis, ...])
        ret = geometry.points_in_convex_polygon_jit(
            gt_boxes_bv.reshape(-1, 2), bounding_box)
        return np.any(ret.reshape(-1, 4), axis=1)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        example_merged = defaultdict(list)
        for example in batch_list:
            for k, v in example.items():
                example_merged[k].append(v)
        ret = {}
        # example_merged.pop("num_voxels")
        for key, elems in example_merged.items():
            if key in ['voxels', 'num_points', 'num_gt', 'voxel_labels', 'match_indices', 'voxel_centers']:
                ret[key] = np.concatenate(elems, axis=0)
            elif key == 'match_indices_num':
                ret[key] = np.concatenate(elems, axis=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key == 'left_img' or key == 'right_img':
                imgs = []
                for img in elems:
                    img = np.transpose(img, [2, 0, 1])
                    imgs.append(img)
                ret[key] = np.stack(imgs, axis=0)
            elif key == 'gt_boxes':
                max_gt = 0
                batch_size = elems.__len__()
                for k in range(batch_size):
                    max_gt = max(max_gt, elems[k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                ret[key] = batch_gt_boxes3d
            else:
                ret[key] = np.stack(elems, axis=0)
        return ret
