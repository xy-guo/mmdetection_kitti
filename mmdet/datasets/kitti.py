import os

import mmcv
import numpy as np
from mmcv.utils import print_log
import logging
from collections import OrderedDict

from mmdet.core import eval_recalls
from mmdet.utils.debug import is_debug
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class KittiDataset(CustomDataset):

    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        self.cat_ids = self.CLASSES
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        metas = mmcv.load(ann_file)
        if is_debug('FASTEPOCH'):
            metas = metas[:64]
        img_infos = []
        for meta in metas:
            img_info = dict()
            if 'annos' in meta:
                img_info["annos"] = meta["annos"]
            img_info["cam_params"] = self.reformat_transformation_matrices(
                meta)
            img_info["image_idx"] = meta["image_idx"]
            img_info["filename"] = meta["img_path"]
            img_info["left_filename"] = meta["img_path"]
            img_info["right_filename"] = meta["img_path"].replace(
                "image_2", "image_3")
            img_info["velodyne_filename"] = meta["velodyne_path"]
            img_info['width'] = meta["img_shape"][1]
            img_info['height'] = meta["img_shape"][0]
            img_infos.append(img_info)
        return img_infos

    def decompose_kitti_intrinsics(self, P):
        """

        :param P: [4, 4]
        :return: intrinsics K [3, 3], and translation t [3, 1]
        """
        assert P.shape == (4, 4)
        K = P[:3, :3]
        Kt = P[:3, 3:4]
        t = np.linalg.inv(K) @ Kt
        return K, t

    def reformat_transformation_matrices(self, cam_params):
        Trv2c, rect, P2, P3 = cam_params['calib/Tr_velo_to_cam'], cam_params['calib/R0_rect'], cam_params['calib/P2'], cam_params['calib/P3']
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
        # rect = np.eye(4, dtype=np.float32)
        new_cam_params = {
            "velo2cam2": Trv2c2,  # transformation from velo to cam2
            "velo2cam3": Trv2c3,  # transformation from velo to cam3
            "K": K,  # intrinsics
            "t2to3": t2to3,  # translation from cam2 to cam3
            "fL": -K[0, 0] * t2to3[0]  # focal length * rectified translation
        }
        return new_cam_params

    def get_ann_info(self, idx):
        if 'annos' in self.data_infos[idx]:
            return self._parse_ann_info(self.data_infos[idx], self.data_infos[idx]["annos"])
        else:
            return None

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        # TODO: check the correctness, seems useless for kitti
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt:
                anno_names = img_info['annos']['name']
                anno_names = [x for x in anno_names if x in self.CLASSES]
                if len(anno_names) <= 0:
                    continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        print(f"{len(valid_inds)} valid samples in {len(self.data_infos)} img_infos")
        return valid_inds

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        # for k, v in results['img_info']['cam_params'].items():
        # results[k] = v

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation from the raw annotation dict.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels
        """
        gt_bboxes = []
        gt_bboxes_3d = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_3d_ignore = []

        # MIN_HEIGHT = [40, 25, 25]
        # MAX_OCCLUSION = [0, 1, 2]
        # MAX_TRUNCATION = [0.15, 0.3, 0.5]

        for i in range(len(ann_info['name'])):
            # Car, Van, Truck, Pedestrain, Person_sitting, Cyclist, Tram, Misc, DontCare
            cls_name = ann_info['name'][i]
            bbox = ann_info['bbox'][i]  # 0-based [left, top, right, bottom]
            bbox3d = np.concatenate([ann_info['location'][i],
                                     ann_info['dimensions'][i],
                                     [ann_info['rotation_y'][i]]], 0)
            difficulty = ann_info['difficulty'][i]  # 1, 2, 3
            # dimensions = ann_info['dimensions'][i]  # height, width, length
            # location = ann_info['location'][i]  # x, y, z incamera coordinates
            # alpha = ann_info['alpha'][i]  # observation angle [-pi, pi]
            # rotation_y = ann_info['rotation_y'][i]  # rotation along y-axis [-pi, pi]
            # occluded = ann_info['occluded'][i]  # 0,1,2,3; 3 means unkown
            # truncated = ann_info['truncated'][i]  # 0~1, leaving image boundary

            if cls_name in self.CLASSES:
                if difficulty >= 3:
                    # either highly occluded, trucated or the object is too small
                    gt_bboxes_ignore.append(bbox)
                    gt_bboxes_3d_ignore.append(bbox3d)
                else:
                    gt_bboxes.append(bbox)
                    gt_bboxes_3d.append(bbox3d)
                    gt_labels.append(self.cat2label[cls_name])
            else:
                if cls_name == 'Van' and 'Car' in self.CLASSES:
                    gt_bboxes_ignore.append(bbox)
                    gt_bboxes_3d_ignore.append(bbox3d)
                elif cls_name == 'Person_sitting' and 'Pedestrian' in self.CLASSES:
                    gt_bboxes_ignore.append(bbox)
                    gt_bboxes_3d_ignore.append(bbox3d)
                elif cls_name == 'DontCare':
                    gt_bboxes_ignore.append(bbox)
                    gt_bboxes_3d_ignore.append(bbox3d)
                else:
                    if cls_name not in ['Truck', 'Tram', 'Misc']:
                        print(cls_name, 'not handled')


        if self.filter_empty_gt:
            assert len(
                gt_bboxes) > 0, "gt_bboxes num should > 0 when empty gt is filtered"
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            gt_bboxes_3d_ignore = np.array(
                gt_bboxes_3d_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            gt_bboxes_3d_ignore = np.zeros((0, 7), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            bboxes_3d=gt_bboxes_3d,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            bboxes_3d_ignore=gt_bboxes_3d_ignore)
        return ann

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, save_dir=None, save_debug_dir=None, **kwargs):
        """Format the results to txt (standard format for Kitti evaluation).

        Args:
            results (list): Testing results of the dataset.
            save_dir (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the txt filepaths, tmp_dir is the temporal directory created
                for saving txt files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        det_annos = []
        for idx, bbox_list in enumerate(results):
            sample_idx = self.data_infos[idx]['image_idx']
            num_example = 0

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': []}
            for cls_idx, cls_bbox in enumerate(bbox_list):
                cls_name = self.cat_ids[cls_idx]
                bbox_2d_preds = cls_bbox[:, :4]
                scores = cls_bbox[:, 4]
                # TODO: 3d bbox prediction
                for bbox_2d_pred, score in zip(bbox_2d_preds, scores):
                    # TODO: out of range detection, should be filtered in the network part
                    anno["score"].append(score)
                    anno["name"].append(cls_name)
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["bbox"].append(bbox_2d_pred)
                    anno["alpha"].append(0.)
                    anno["dimensions"].append(
                        np.array([0, 0, 0], dtype=np.float32))
                    anno["location"].append(
                        np.array([-1000, -1000, -1000], dtype=np.float32))
                    anno["rotation_y"].append(0.)
                    num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                    'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])}

            anno["sample_idx"] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

            if save_dir is not None:
                cur_det_file = os.path.join(
                    save_dir, 'results', '%06d.txt' % sample_idx)
                print("saving results to", cur_det_file)

                # dump detection results into txt files
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2],
                                 anno['rotation_y'][idx], anno['score'][idx]), file=f)

            if save_debug_dir is not None:
                # dump debug infos into pkl files
                cur_debug_file = os.path.join(
                    save_debug_dir, 'det_info_%06d.pkl' % sample_idx)
                debug_infos = {
                    'anno': anno,
                    'gt_anno': self.data_infos[idx]["annos"],
                }
                with open(cur_debug_file, 'wb') as f:
                    mmcv.dump(debug_infos, f)

        return det_annos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 txtfile_prefix=None,
                 proposal_nums=(100, 300, 1000)):
        if 'annos' not in self.data_infos[0]:
            print_log('The testing results of the whole dataset is empty.',
                      logger=logger, level=logging.ERROR)
            raise ValueError(
                "annotations not available for the test set of KITTI")

        print_log('Evaluating KITTI object detection \n', logger=logger)
        det_annos = self.format_results(results, txtfile_prefix)
        gt_annos = [x['annos'] for x in self.data_infos]

        from mmdet.datasets.kitti_object_eval_python.eval import get_official_eval_result
        eval_results = get_official_eval_result(
            gt_annos, det_annos, self.CLASSES)

        return_results = OrderedDict()
        for cls_name, ret in eval_results.items():
            for k, v in ret.items():
                msg = ','.join([f"{vv:.3f}" for vv in v])
                return_results[f"{cls_name}_{k}"] = msg
                print_log(f"{cls_name}_{k}:  {msg}", logger=logger)

        return return_results
