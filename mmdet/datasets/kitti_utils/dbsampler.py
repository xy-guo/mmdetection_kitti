import numpy as np
import copy
import os
import numpy as np
import utils.box_np_ops as box_np_ops
import datasets.data_augment_utils as data_augment_utils


class BatchSampler:
    def __init__(self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler(object):
    def __init__(self, db_infos, sampler_cfg, logger=None):
        super().__init__()

        if logger is not None:
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")
        for prep_func, val in sampler_cfg.PREPARE.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        if logger is not None:
            logger.info("After filter database:")
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self.rate = sampler_cfg.RATE
        self.sample_groups = []
        for x in sampler_cfg.SAMPLE_GROUPS:
            name, num = x.split(':')
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)

        self.object_rot_range = sampler_cfg.OBJECT_ROT_RANGE
        self.object_rot_enable = np.abs(self.object_rot_range[0] - self.object_rot_range[1]) >= 1e-3

        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info["difficulty"] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, root_path, gt_boxes, gt_names, road_planes, rect=None, Trv2c=None,
                   P2=None, random_crop=False):
        """ do sampling

        :param root_path: dataset root path
        :param gt_boxes: gt boxes in lidar coordinates
        :param gt_names: gt class names
        :param road_planes: road planes in the camera coordinate
        :param rect: rect
        :param Trv2c: velo to cam
        :param P2: P2
        :param random_crop: do random cropping
        :return: sampled point cloud
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes, self.sample_max_nums):
            sampled_num = int(max_sample_num - np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                # the number of sampled point cloud subsets may be less than sampled_num due to collision
                sampled_cls = self.sample_class_v2(class_name, sampled_num, avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack([s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)  # in the lidar coordinate
            center = sampled_gt_boxes[:, 0:3]  # in the lidar coordinate

            if road_planes is not None:
                # image plane
                a, b, c, d = road_planes  # in the image coordinate
                cur_height = (- d - a * center[:, 0] - c * center[:, 1]) / b  # estimate z, x from lidar coordinates x,y
                tmp_point = np.zeros((len(cur_height), 3))
                tmp_point[:, 1] = cur_height
                lidar_point = box_np_ops.camera_to_lidar(tmp_point, rect, Trv2c)
                cur_lidar_height = lidar_point[:, 2]
                mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                sampled_gt_boxes[:, 2] -= mv_height  # camera view

            num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(root_path, info["path"])
                s_points = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])

                if "rot_transform" in info:
                    rot = info["rot_transform"]
                    s_points[:, :3] = box_np_ops.rotation_points_single_angle(s_points[:, :3], rot, axis=2)
                s_points[:, :3] += info["box3d_lidar"][:3]
                if road_planes is not None:
                    # mv height
                    s_points[:, 2] -= mv_height[count]
                count += 1

                s_points_list.append(s_points)

            # do random crop.
            if random_crop:
                assert False, 'TODO: no random crop currently'

            ret = {"gt_names": np.array([s["name"] for s in sampled]),
                   "difficulty": np.array([s["difficulty"] for s in sampled]), "gt_boxes": sampled_gt_boxes,
                   "points": np.concatenate(s_points_list, axis=0), "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
                   "group_ids": np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))}
        # return sampled results
        return ret

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate([valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self.object_rot_enable:
            assert False, 'This part needs to be checked'
            # place samples to any place in a circle.
            data_augment_utils.noise_per_object_v3_(boxes, None, valid_mask, 0, 0, self._global_rot_range, num_try=100)

        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self.object_rot_enable:
                    assert False, 'This part needs to be checked'
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (boxes[i, -1] - sp_boxes[i - num_gt, -1])
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
