import numpy as np
import torch
import torch.nn.functional as F
import mmcv
import cv2

from mmdet.core import bbox2result, bbox3d2result, bbox2roi, build_assigner, build_sampler
from mmdet.core.bbox.geometry import bbox_overlaps as bbox_overlaps_2d
from .. import builder
from ..registry import DETECTORS
# from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from .base import StereoBaseDetector
from mmdet.utils.det3d import box_np_ops
from mmdet.core import get_classes, tensor2imgs


@DETECTORS.register_module
class StereoDsgnSingleStageDetector(StereoBaseDetector):

    def __init__(self,
                 backbone=None,
                 stereo_neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 bbox3d_roi_extractor=None,
                 bbox3d_head=None,
                 bbox3d_block_extractor=None,
                 bbox3d_refine_head=None,
                 depth_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_raw_proposals_for_3d=False,
                 expand_3droi_as_square=False):
        super(StereoDsgnSingleStageDetector, self).__init__(
            backbone=backbone,
            stereo_backbone=stereo_backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if bbox3d_head is not None:
            self.bbox3d_roi_extractor = builder.build_roi_extractor(
                bbox3d_roi_extractor)
            self.depth_roi_extractor = builder.build_roi_extractor(
                depth_roi_extractor)
            self.bbox3d_head = builder.build_head(bbox3d_head)

            if debug:
                tmp = bbox_roi_extractor.copy()
                tmp["out_channels"] = 3
                self.bbox_roi_extractor_pyramid = builder.build_roi_extractor(
                    tmp)
                tmp = bbox3d_roi_extractor.copy()
                tmp["out_channels"] = 3
                self.bbox3d_roi_extractor_pyramid = builder.build_roi_extractor(
                    tmp)

        if bbox3d_refine_head is not None:
            self.bbox3d_block_extractor = builder.build_roi_extractor(
                bbox3d_block_extractor)
            self.bbox3d_refine_head = builder.build_head(bbox3d_refine_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.debug = debug
        self.debug_use_gt_volume = debug_use_gt_volume
        self.debug_use_gt_2d_bbox = debug_use_gt_2d_bbox
        self.debug_always_train_2d = debug_always_train_2d
        self.use_raw_proposals_for_3d = use_raw_proposals_for_3d
        self.expand_3droi_as_square = expand_3droi_as_square

        if self.with_mask:
            raise NotImplementedError
        if not self.with_rpn or not self.with_bbox:
            raise NotImplementedError

        self.init_weights(pretrained=pretrained)

    @property
    def with_bbox3d(self):
        return hasattr(self, 'bbox3d_head') and self.bbox3d_head is not None

    @property
    def with_bbox3d_refine(self):
        return hasattr(
            self, 'bbox3d_refine_head') and self.bbox3d_refine_head is not None

    def init_weights(self, pretrained=None):
        super(StereoThreeStageDebugDetector, self).init_weights(pretrained)
        if self.with_bbox3d:
            self.bbox3d_roi_extractor.init_weights()
            self.bbox3d_head.init_weights()

    def forward_train(self,
                      left_img,
                      right_img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_bboxes_3d=None,
                      gt_bboxes_3d_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      depth_img=None,
                      lidar_points=None,
                      uvd_points=None,
                      proposals3d=None,
                      **cam_params):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert proposals3d is None

        left_x = self.extract_feat(left_img)
        right_x = self.extract_feat(right_img)

        if self.stereo_backbone is not None:
            left_stereo_x = self.extract_stereo_feat(left_img)
            right_stereo_x = self.extract_stereo_feat(right_img)
        else:
            left_stereo_x = left_x
            right_stereo_x = right_x

        if False and self.debug:
            with torch.no_grad():
                left_pyramid = [
                    F.avg_pool2d(left_img, s, stride=s)
                    for s in [4, 8, 16, 32, 64]
                ]
                right_pyramid = [
                    F.avg_pool2d(right_img, s, stride=s)
                    for s in [4, 8, 16, 32, 64]
                ]

        losses = dict()

        if (not self.debug_use_gt_2d_bbox) or self.debug_always_train_2d:
            # RPN forward and loss
            # rpn_outs: tuple ([List lvl: B-Na-Hi-Wi tensor], [List lvl: B-4Na-Hi-Wi tensor])
            # rpn_losses: dict {loss_rpn_bbox: [List lvl: scalar], loss_rpn_cls: [List lvl: scalar]}
            # proposal_list: [List: N_proposal-5 tensor]
            if self.with_rpn:
                rpn_outs = self.rpn_head(left_x)
                rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                              self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses.update(rpn_losses)

                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
                proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            else:
                proposal_list = proposals

            # The assign process will assign gt bboxes and labels to the proposals
            # AssignResult is a class with the following keys:
            #       assigned_gt_inds: [N_proposal] tensor.
            #           not assigned is -1, negative sample is 0, positive sample is gt_box_id + 1
            #       labels: [N_proposal] tensor. assigned labels
            #           not assigned & negative is 0, positive sample is 1-based label id
            #       max_overlaps: [N_proposal] tensor. maximum overlap with gt bbox
            #       num_gts: N_gt int. the number of gt bboxes
            #
            # The sampling process will first concat gt_bboxes to all proposal bboxes.
            # SampingResult is a class with the following keys:
            #       bboxes: [N_proposal+N_gt, 4] tensor, used for roi-pooling
            #           NOTE: it is pos_bboxes + neg_bboxes after __init__, not the input argument
            #       gt_bboxes: [N_gt, 4] tensor, used for computing loss
            #       gt_flags: [N_proposal+N_gt] tensor, indicate whether the bboxes are copied from gt
            #           [1, 1, 1, ..., 1, 0, 0, 0, ..., 0]
            #       assign_result: The gt labels are added to assign_result
            #           The corresponding assigned_gt_inds, labels, max_overlaps are updated
            #                             [1, 2, ...]       [gt_labels]  [1.0, 1.0, ...]
            #       pos_inds: [N_sampled_pos] tensor (sampled from bboxes)
            #           if positive samples in AssignResults are small, then use all of the positive assigned samples
            #       pos_bboxes: [N_sampled_pos, 4]
            #       pos_is_gt: [N_sampled_pos]  sampled positive samples are gt or not?
            #       pos_assigned_gt_inds: [N_sampled_pos]  corresponding gt indices
            #       pos_gt_bboxes / pos_gt_labels: subset of pos_bboxes which are from gt
            #       neg_inds: [N_sampled_neg] tensor (sampled from bboxes)
            #       neg_bboxes: [N_sampled_neg, 4]

            if self.with_bbox:
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
                num_imgs = left_img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in left_x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            # rois: [N_roi, 5] tensor. for each roi: b, x1, y1, x2, y2
            # bbox_feats: [N_roi, 256, 7, 7]
            # cls_score: [N_roi, N_classes]
            # bbox_pred: [N_roi, 4 * N_classes]
            # bbox_targets: labels, label_weights, bbox_targets, bbox_weights
            if self.with_bbox:
                print("    rcnn pos/neg:",
                      [len(res.pos_bboxes) for res in sampling_results],
                      [len(res.neg_bboxes) for res in sampling_results])
                assert all([len(res.bboxes) > 0
                            for res in sampling_results]), \
                    "no bbox for rcnn: pos {} / neg {}".format(
                    [len(res.pos_bboxes) for res in sampling_results],
                    [len(res.neg_bboxes) for res in sampling_results])
                rois = bbox2roi([res.bboxes for res in sampling_results])
                # TODO: a more flexible way to decide which feature maps to use
                bbox_feats = self.bbox_roi_extractor(
                    left_x[:self.bbox_roi_extractor.num_inputs], rois)

                if False and self.debug:
                    with torch.no_grad():
                        bbox_imgs = self.bbox_roi_extractor_pyramid(
                            left_pyramid[:self.bbox_roi_extractor.num_inputs],
                            rois)
                        mean = bbox_imgs.new_tensor([123.675, 116.28,
                                                     103.53]).view(1, 3, 1, 1)
                        std = bbox_imgs.new_tensor([58.395, 57.12,
                                                    57.375]).view(1, 3, 1, 1)
                        bbox_imgs = (bbox_imgs * std) + mean
                        left_img_np = (left_img * std) + mean
                        for idx, img_i in enumerate(left_img_np):
                            img_i = img_i.permute(
                                1, 2, 0).detach().cpu().numpy().astype(
                                    np.uint8)[:, :, [2, 1, 0]]
                            mmcv.imwrite(
                                img_i,
                                f"work_dir/debug/proposals/img_{idx}.jpg")
                        for idx, img_i in enumerate(bbox_imgs):
                            img_i = img_i.permute(
                                1, 2, 0).detach().cpu().numpy().astype(
                                    np.uint8)[:, :, [2, 1, 0]]
                            mmcv.imwrite(
                                img_i, f"work_dir/debug/proposals/{idx}.jpg")

                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = self.bbox_head(bbox_feats)

                bbox_targets = self.bbox_head.get_target(
                    sampling_results, gt_bboxes, gt_labels,
                    self.train_cfg.rcnn)
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                                *bbox_targets)
                losses.update(loss_bbox)

        if self.with_bbox3d and proposals3d is None:

            def filter_by_area(bboxes):
                x1, y1, x2, y2 = bboxes[:, 0], bboxes[:,
                                                      1], bboxes[:,
                                                                 2], bboxes[:,
                                                                            3]
                areas = (x2 - x1) * (y2 - y1)
                valid_inds = areas > 25
                print("    filter {} in {}".format(valid_inds.sum(),
                                                   len(valid_inds)))
                return bboxes[valid_inds]

            assert gt_bboxes_3d is not None, "no 3d gt bboxes inputs"

            if not self.debug_use_gt_2d_bbox:
                # refined_bboxes_list is the refine 2d bboxes for each image
                with torch.no_grad():
                    proposal_list_refined = self.bbox_head.refine_bboxes(
                        rois,
                        bbox_targets[0],  # 0 is the labels
                        bbox_pred,
                        pos_is_gts=[res.pos_is_gt for res in sampling_results],
                        img_metas=img_meta)
                if self.use_raw_proposals_for_3d:
                    proposal_list_refined = [
                        torch.cat([xx[:, :4], yy[:, :4]], 0)
                        for xx, yy in zip(proposal_list, proposal_list_refined)
                    ]
                proposal_list_refined = [
                    filter_by_area(xx).detach() for xx in proposal_list_refined
                ]
            else:
                # NOTE: if debug_use_gt_2d_bbox=True, use ground truth 2d bboxes
                proposal_list_refined = [x.clone().detach() for x in gt_bboxes]
                sampling_results = None

            # assign and sample again
            # concatenate the gt bboxes, however, we need sampling results to provide ?
            bbox_assigner = build_assigner(self.train_cfg.rcnn3d.assigner3d)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn3d.sampler3d, context=self)
            num_imgs = left_img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results_2d = sampling_results
            sampling_results = []
            for i in range(num_imgs):
                # assign ground truth ids for each proposals
                assign_result = bbox_assigner.assign(proposal_list_refined[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                #
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list_refined[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_to_be_used=gt_bboxes_3d[i],
                    feats=[lvl_feat[i][None] for lvl_feat in left_x])
                # NOTE: in sampling results, bbox is 2d and gt is 3d
                # TODO(xyg): add support add_gt_as_proposals when 2d-3d case
                sampling_results.append(sampling_result)

            if self.expand_3droi_as_square:

                def make_square(bboxes):
                    widths = bboxes[:, 2] - bboxes[:, 0]  # x2 - x1
                    heights = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
                    pad_widths = torch.clamp(heights - widths, min=0.) / 2
                    pad_heights = torch.clamp(widths - heights, min=0.) / 2
                    new_bboxes = torch.stack([
                        bboxes[:, 0] - pad_widths, bboxes[:, 1] - pad_heights,
                        bboxes[:, 2] + pad_widths, bboxes[:, 3] + pad_heights
                    ],
                                             dim=1)
                    return new_bboxes

                for res in sampling_results:
                    res.pos_bboxes = make_square(res.pos_bboxes)
                    res.neg_bboxes = make_square(res.neg_bboxes)

            print("    3d pos/neg:",
                  [len(res.pos_bboxes) for res in sampling_results],
                  [len(res.neg_bboxes) for res in sampling_results])
            assert all([len(res.bboxes) > 0
                        for res in sampling_results]), \
                "no bbox for 3d: pos {} / neg {}, corresponding rcnn pos {} / neg {}".format(
                [len(res.pos_bboxes) for res in sampling_results],
                [len(res.neg_bboxes) for res in sampling_results],
                [len(res.pos_bboxes) for res in sampling_results_2d] if sampling_results_2d is not None else None,
                [len(res.neg_bboxes) for res in sampling_results_2d] if sampling_results_2d is not None else None)
            # 2d rois
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # print("num of rois", [
            #   res.bboxes.shape for res in sampling_results])
            depth_levels = self.bbox3d_head.get_depth_levels(
                [res.bboxes for res in sampling_results],
                self.train_cfg.rcnn3d)

            bbox_feats3d = self.bbox3d_roi_extractor(
                left_stereo_x[:self.bbox_roi_extractor.num_inputs],
                right_stereo_x[:self.bbox_roi_extractor.num_inputs], rois,
                torch.cat(depth_levels, 0), cam_params["K"], cam_params["fL"])

            depth_gts = self.depth_roi_extractor(depth_img, rois)

            if self.debug_use_gt_volume:
                with torch.no_grad():
                    disp_sample = torch.cat(depth_levels,
                                            0).unsqueeze(-1).unsqueeze(-1)
                    distance = torch.abs(disp_sample - depth_gts)
                    clamp_dist = 1.6
                    volume = clamp_dist - distance.clamp(max=clamp_dist)
                    volume[torch.isnan(volume)] = 0.
                    bbox_feats3d['volume'] = volume.unsqueeze(1).repeat(
                        1, 256, 1, 1, 1)

            bbox3d_head_pred = self.bbox3d_head(bbox_feats3d)

            bbox_targets3d = self.bbox3d_head.get_target(
                sampling_results, gt_bboxes_3d, gt_labels, depth_levels,
                cam_params["K"], self.train_cfg.rcnn3d)
            bbox_targets3d['depth_gt'] = depth_gts
            loss_bbox3d = self.bbox3d_head.loss(
                bbox3d_head_pred, bbox_targets3d, depth_levels=depth_levels)

            losses.update(loss_bbox3d)

            if self.debug:
                with torch.no_grad():
                    # show sampling_result to check errors
                    pos_proposals = [
                        res.pos_bboxes for res in sampling_results
                    ]
                    # 2d neg proposals
                    neg_proposals = [
                        res.neg_bboxes for res in sampling_results
                    ]
                    # ground truth 3d bboxes
                    pos_gt_bboxes3d = [
                        res.pos_gt_bboxes for res in sampling_results
                    ]
                    # ground truth labels
                    pos_gt_labels = [
                        res.pos_gt_labels for res in sampling_results
                    ]

                    mean = left_img.new_tensor([123.675, 116.28,
                                                103.53]).view(1, 3, 1, 1)
                    std = left_img.new_tensor([58.395, 57.12,
                                               57.375]).view(1, 3, 1, 1)
                    imgs = (left_img * std) + mean
                    imgs = imgs.permute(0, 2, 3, 1)

                    def to_numpy(x):
                        return x.detach().cpu().numpy()

                    def list_to_numpy(x):
                        return [to_numpy(xx) for xx in x]

                    imgs = to_numpy(imgs)
                    pos_proposals = list_to_numpy(pos_proposals)
                    neg_proposals = list_to_numpy(neg_proposals)
                    pos_gt_bboxes3d = list_to_numpy(pos_gt_bboxes3d)
                    pos_gt_labels = list_to_numpy(pos_gt_labels)
                    bboxes_ignore = list_to_numpy(gt_bboxes_ignore)

                    class_names = ('Car', 'Ped', "Cyc")
                    for idx in range(len(imgs)):
                        # print("idx: ", idx, len(pos_proposals[idx]),
                        #       "pos", len(neg_proposals[idx]), "neg")
                        # print('pos_proposals', pos_proposals)
                        img = imgs[idx].copy()[..., [2, 1, 0]].copy().astype(
                            np.uint8)
                        K = to_numpy(cam_params['K'][idx])
                        mmcv.imshow_det_bboxes(
                            img,
                            pos_proposals[idx],
                            pos_gt_labels[idx] - 1,
                            class_names=class_names,
                            bbox_color='red',
                            text_color='red',
                            show=False)
                        mmcv.imshow_det_bboxes(
                            img,
                            neg_proposals[idx],
                            np.zeros([neg_proposals[idx].shape[0]],
                                     dtype=np.int),
                            class_names=[''],
                            bbox_color='cyan',
                            text_color='cyan',
                            show=False)
                        mmcv.imshow_det_bboxes(
                            img,
                            bboxes_ignore[idx],
                            np.zeros([bboxes_ignore[idx].shape[0]],
                                     dtype=np.int),
                            class_names=[''],
                            bbox_color='blue',
                            text_color='blue',
                            show=False)
                        # imshow_3d_det_bboxes(
                        #     img,
                        #     get_corners_from_3d_bboxes(
                        #         pos_gt_bboxes3d[idx], K),
                        #     pos_gt_labels[idx] - 1,
                        #     class_names=class_names,
                        #     show=False,
                        #     bbox_color='green',
                        #     text_color='green',
                        #     out_file=None,
                        #     random_color=False)
                        imshow_3d_det_bboxes(
                            img,
                            get_corners_from_3d_bboxes(
                                gt_bboxes_3d[idx].detach().cpu().numpy(), K),
                            gt_labels[idx].detach().cpu().numpy() - 1,
                            class_names=class_names,
                            show=False,
                            bbox_color='green',
                            text_color='green',
                            out_file=None,
                            random_color=False)

                        import os
                        existed_files = sorted(
                            os.listdir('work_dir/debug/check_proposals/'))
                        if len(existed_files) > 0:
                            previous_out_idx = [
                                int(fname.split('_')[1].split('.')[0])
                                for fname in existed_files
                            ]
                            out_idx = sorted(previous_out_idx)[-1] + 1
                        else:
                            out_idx = 1
                        mmcv.imwrite(
                            img,
                            'work_dir/debug/check_proposals/debug_{}_{}.jpg'.
                            format(out_idx, idx))

            if False and self.debug:
                with torch.no_grad():
                    results = self.bbox3d_roi_extractor_pyramid(
                        left_pyramid[:self.bbox_roi_extractor.num_inputs],
                        right_pyramid[:self.bbox_roi_extractor.num_inputs],
                        rois, torch.cat(depth_levels,
                                        0), cam_params["K"], cam_params["fL"])
                    bbox_imgs, bbox_imgs_l = results["volume"], results[
                        "img_feature"]
                    bbox_imgs_l = bbox_imgs_l.unsqueeze(2).repeat(
                        1, 1, bbox_imgs.shape[2], 1, 1)
                    mean = bbox_imgs.new_tensor([123.675, 116.28,
                                                 103.53]).view(1, 3, 1, 1, 1)
                    std = bbox_imgs.new_tensor([58.395, 57.12,
                                                57.375]).view(1, 3, 1, 1, 1)
                    bbox_imgs = (bbox_imgs * std) + mean
                    bbox_imgs_l = (bbox_imgs_l * std) + mean
                    for idx, (img_i, img_i_l) in enumerate(
                            zip(bbox_imgs, bbox_imgs_l)):
                        reg_weight = bbox_targets3d['regression_weights'][
                            idx, ..., 0].sum(-1)
                        if idx < 100:
                            # img_i: [3, d, h, w]
                            img_i = img_i.view(3, -1, img_i.shape[-1])
                            img_i = img_i.permute(
                                1, 2, 0).detach().cpu().numpy().astype(
                                    np.uint8)[:, :, [2, 1, 0]]
                            img_i_l *= reg_weight.view(1, -1, 1, 1) * 0.7 + 0.3
                            img_i_l = img_i_l.view(3, -1, img_i_l.shape[-1])
                            img_i_l = img_i_l.permute(
                                1, 2, 0).detach().cpu().numpy().astype(
                                    np.uint8)[:, :, [2, 1, 0]]
                            mmcv.imwrite(
                                np.concatenate([img_i, img_i_l], 1),
                                f"work_dir/debug/proposals3d/{idx}.jpg")
        if self.with_bbox3d_refine:
            assert gt_bboxes_3d is not None, "no 3d gt bboxes inputs"
            assert proposals3d is not None, "no proposals3d provided"
            gt_bboxes_3d = [x[:, :7] for x in gt_bboxes_3d]
            proposal_list_3d = proposals3d

            # assign and sample again
            # concatenate the gt bboxes, however, we need sampling results to provide ?
            bbox_assigner = build_assigner(
                self.train_cfg.refine3d.assigner_prerefine)
            bbox_sampler = build_sampler(
                self.train_cfg.refine3d.sampler_prerefine, context=self)
            num_imgs = left_img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # assign ground truth ids for each proposals
                assign_result = bbox_assigner.assign(proposal_list_3d[i],
                                                     gt_bboxes_3d[i], None,
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list_3d[i],
                    gt_bboxes_3d[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in left_x])
                # NOTE: in sampling results, bbox is 2d and gt is 3d
                # TODO(xyg): add support add_gt_as_proposals when 2d-3d case
                sampling_results.append(sampling_result)

            sampled_bboxes3d = [res.bboxes for res in sampling_results]
            sampled_pos_is_gt = [res.pos_is_gt for res in sampling_results]
            sampled_labels = [res.pos_gt_labels for res in sampling_results]
            assert all([len(xx) > 0 for xx in sampled_bboxes3d
                        ]), "at least one sample for each image"
            bbox_feats3d = self.bbox3d_block_extractor(
                left_x[:self.bbox_roi_extractor.num_inputs],
                right_x[:self.bbox_roi_extractor.num_inputs], sampled_bboxes3d,
                cam_params["K"], cam_params["t2to3"], cam_params["fL"])
            # depth_gts = self.depth_roi_extractor(depth_img, rois)

            cls_scores, bbox_preds = self.bbox3d_refine_head(bbox_feats3d)
            loss_bbox3d = self.bbox3d_refine_head.loss(
                sampled_bboxes3d,
                sampled_pos_is_gt,
                sampled_labels,
                cls_scores,
                bbox_preds,
                gt_bboxes_3d,
                gt_labels,
                img_meta,
                self.train_cfg.refine3d,
                gt_bboxes3d_ignore=gt_bboxes_3d_ignore)
            losses.update(loss_bbox3d)

        return losses

    def convert_roi_depths_to_3d_points(self, depth_maps, K, rois):
        assert len(depth_maps.shape) == 3
        assert K.shape == (3, 3)
        uv = self.depth_roi_extractor.get_roi_image_coordinates(rois)
        uvd = torch.cat(
            [uv * depth_maps.unsqueeze(-1),
             depth_maps.unsqueeze(-1)], dim=-1)
        xyz = uvd @ torch.inverse(K).transpose(1, 0)
        return xyz.view(-1, 3)

    def simple_test_3d_bboxes(self,
                              left_x,
                              right_x,
                              img_meta,
                              proposals,
                              K,
                              fL,
                              rcnn_test_cfg,
                              depth_img,
                              rescale=False,
                              labels_2d=None,
                              scores_2d=None):
        """Test only det bboxes without augmentation."""
        if self.expand_3droi_as_square:

            def make_square(bboxes):
                widths = bboxes[:, 2] - bboxes[:, 0]  # x2 - x1
                heights = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
                pad_widths = torch.clamp(heights - widths, min=0.) / 2
                pad_heights = torch.clamp(widths - heights, min=0.) / 2
                new_bboxes = torch.stack([
                    bboxes[:, 0] - pad_widths, bboxes[:, 1] - pad_heights,
                    bboxes[:, 2] + pad_widths, bboxes[:, 3] + pad_heights
                ],
                                         dim=1)
                return new_bboxes

            proposals = [make_square(xx) for xx in proposals]
        rois = bbox2roi(proposals)
        # TODO(xyg): test cfg for get depth levels?
        depth_levels = self.bbox3d_head.get_depth_levels(proposals, None)
        roi_feats3d = self.bbox3d_roi_extractor(
            left_x[:len(self.bbox_roi_extractor.featmap_strides)],
            right_x[:len(self.bbox_roi_extractor.featmap_strides)], rois,
            torch.cat(depth_levels, 0), K, fL)
        roi_depth_gts = self.depth_roi_extractor(depth_img, rois).squeeze(1)
        if self.debug_use_gt_volume:
            with torch.no_grad():
                disp_sample = torch.cat(depth_levels,
                                        0).unsqueeze(-1).unsqueeze(-1)
                distance = torch.abs(disp_sample - roi_depth_gts.unsqueeze(1))
                clamp_dist = 1.6
                volume = clamp_dist - distance.clamp(max=clamp_dist)
                volume[torch.isnan(volume)] = 0.
                roi_feats3d['volume'] = volume.unsqueeze(1).repeat(
                    1, 256, 1, 1, 1)
        bbox_head_pred3d = self.bbox3d_head(roi_feats3d)
        roi_depth_preds = self.bbox3d_head.get_abs_depth_from_logits(
            bbox_head_pred3d['abs_depth_pred'], depth_levels)
        roi_points_gts = self.convert_roi_depths_to_3d_points(
            roi_depth_gts, K[0], rois)
        roi_points_preds = self.convert_roi_depths_to_3d_points(
            roi_depth_preds, K[0], rois)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_results = self.bbox3d_head.get_3d_det_bboxes(
            rois,
            bbox_head_pred3d,
            depth_levels[0],  # assume only one image
            K[0],
            fL[0],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg,
            det_labels_2d=labels_2d,
            det_scores_2d=scores_2d)
        return det_results, {
            "roi_points_gts": roi_points_gts,
            "roi_points_preds": roi_points_preds
        }

    def simple_refine_3d_bboxes(self,
                                left_stereo_feats,
                                right_stereo_feats,
                                img_meta,
                                proposals3d,
                                K,
                                fL,
                                t2to3,
                                cfg,
                                rescale=False):
        bbox_feats3d = self.bbox3d_block_extractor(
            left_stereo_feats[:self.bbox_roi_extractor.num_inputs],
            right_stereo_feats[:self.bbox_roi_extractor.num_inputs],
            proposals3d, K, t2to3, fL)
        cls_scores, bbox_preds = self.bbox3d_refine_head(bbox_feats3d)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_results = self.bbox3d_refine_head.get_3d_det_bboxes(
            cls_scores,
            bbox_preds,
            proposals3d[0],
            K[0],
            fL[0],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=cfg)
        return det_results, {}

    def simple_test(
            self,
            left_img,
            right_img,
            img_meta,
            proposals=None,
            rescale=False,
            depth_img=None,
            proposals3d=None,
            gt_bboxes=None,  # only for debug mode
            gt_labels=None,  # only for debug mode
            **cam_params):
        """Test without augmentation."""
        left_feats = self.extract_feat(left_img)
        right_feats = self.extract_feat(right_img)

        if self.stereo_backbone is not None:
            left_stereo_feats = self.extract_stereo_feat(left_img)
            right_stereo_feats = self.extract_stereo_feat(right_img)
        else:
            left_stereo_feats = left_feats
            right_stereo_feats = right_feats

        if (not self.debug_use_gt_2d_bbox) or self.debug_always_train_2d:
            # rpn, ~1000 proposals after 1st-stage nms
            if proposals is None:
                proposal_list = self.simple_test_rpn(left_feats, img_meta,
                                                     self.test_cfg.rpn)
            else:
                proposal_list = proposals

            # refine proposals & the 2nd classfication, multi-class nms
            # det_bboxes: [n_det, 5], bbox + score
            # det_labels: [n_det], 0-based labels
            # TODO: disable  rescale  here
            det_bboxes_with_scores, det_labels = self.simple_test_bboxes(
                left_feats,
                img_meta,
                proposal_list,
                self.test_cfg.rcnn,
                rescale=False)
            det_bboxes = det_bboxes_with_scores[:, :4]
            det_scores = det_bboxes_with_scores[:, 4:5]

            # compute the bbox2d_results with rpn2d & bbox2d networks
            # case1. not in debug mode, always compute the bbox2d_results
            # case2. in debug mode, but 2d part is still trained
            if rescale:
                scale_factor = img_meta[0]['scale_factor']
                assert isinstance(scale_factor, float)
                det_bboxes_with_scores[:, :4] /= scale_factor
            bbox2d_results = bbox2result(det_bboxes_with_scores, det_labels,
                                         self.bbox_head.num_classes)

        if self.debug_use_gt_2d_bbox:
            # synthesize: det_bboxes_with_scores, det_labels, det_scores
            # TODO: pick the 1st item in forward_test
            det_bboxes = gt_bboxes[0][0]
            # TODO: explore when it is 1-based and when 0-based
            det_labels = gt_labels[0][0] - 1
            det_scores = det_bboxes.new_ones([det_bboxes.shape[0], 1])
            det_bboxes_with_scores = torch.cat([det_bboxes, det_scores], dim=1)

            # if rpn2d & bbox2d are not trained, use gt as the final bbox2d_results
            if not self.debug_always_train_2d:
                if rescale:
                    scale_factor = img_meta[0]['scale_factor']
                    assert isinstance(scale_factor, float)
                    det_bboxes_with_scores[:, :4] /= scale_factor
                bbox2d_results = bbox2result(det_bboxes_with_scores,
                                             det_labels,
                                             self.bbox_head.num_classes)

        if proposals3d is None and len(det_bboxes) > 0 and self.with_bbox3d:
            det_results3d, debug_outputs = self.simple_test_3d_bboxes(
                left_stereo_feats,
                right_stereo_feats,
                img_meta, [det_bboxes],
                cam_params["K"],
                cam_params["fL"],
                self.test_cfg.rcnn3d,
                depth_img,
                rescale=rescale,
                labels_2d=det_labels,
                scores_2d=det_scores)
            det_bboxes_3d, det_labels_3d = det_results3d["nms_bboxes3d"]
        else:
            debug_outputs = {}
            det_bboxes_3d = left_img.new_zeros([0, 8])
            det_labels_3d = left_img.new_zeros([0], dtype=torch.int64)

        if self.with_bbox3d_refine and proposals3d is not None and proposals3d[
                0].shape[0] > 0:
            valid_inds = proposals3d[0][:, 3:6].min(1).values > 1.0
            proposals3d[0] = proposals3d[0][valid_inds]
            with torch.no_grad():
                det_results3d, debug_outputs = self.simple_refine_3d_bboxes(
                    left_stereo_feats,  # TODO(xyg): bbox3d refine code is deprecated, verify whether to use left_feats or left_stereo_feats
                    right_stereo_feats,
                    img_meta,
                    proposals3d,
                    cam_params['K'],
                    cam_params['fL'],
                    cam_params['t2to3'],
                    self.test_cfg.refine3d,
                    rescale=rescale)
                det_bboxes_3d, det_labels_3d = det_results3d["nms_bboxes3d"]
            raise NotImplementedError(
            )  # may be not finished, pls double check
        # else:
        #     debug_outputs = {}
        #     det_bboxes_3d = left_img.new_zeros([0, 8])
        #     det_labels_3d = left_img.new_zeros([0], dtype=torch.int64)

        bbox3d_results = bbox3d2result(
            det_bboxes_3d, det_labels_3d,
            self.bbox_head.num_classes)  # results by classes

        reprojected_bboxes_2d = []
        for cls_bboxes_3d in bbox3d_results:
            corners = box_np_ops.center_to_corner_box3d(
                cls_bboxes_3d[:, :3],
                cls_bboxes_3d[:, 3:6],
                cls_bboxes_3d[:, 6],
                origin=[0.5, 1.0, 0.5],
                axis=1)
            corners_2d = corners @ cam_params['K'][0].detach().cpu().numpy().T
            corners_2d = corners_2d[..., :2] / corners_2d[..., 2:3]
            if rescale:
                scale_factor = img_meta[0]['scale_factor']
                assert isinstance(scale_factor, float)
                corners_2d /= scale_factor
            x1y1, x2y2 = corners_2d.min(1), corners_2d.max(1)
            img_h, img_w = img_meta[0]['img_shape'][:2]
            x1y1 = np.maximum(x1y1, 0)
            x2y2 = np.minimum(x2y2, (img_w, img_h))
            reproj_bboxes_2d = np.concatenate(
                [x1y1, x2y2, cls_bboxes_3d[:, 7:8]], 1)
            reprojected_bboxes_2d.append(reproj_bboxes_2d)

        # for idx in range(len(bbox3d_results)):
        #     proposals = det_bboxes[det_labels == idx][:, :4]
        #     proj_bboxes3d = proposals.new_tensor(
        #         reprojected_bboxes_2d[idx][:, :4])
        #     if len(proj_bboxes3d) <= 0 or len(proposals) <= 0:
        #         continue
        #     overlaps = bbox_overlaps_2d(proj_bboxes3d, proposals).max(1)[0]
        #     valid_inds = (overlaps > 0.5).detach(
        #     ).cpu().numpy().astype(np.bool)
        #     print("filter {} / {}".format(valid_inds.sum(), len(valid_inds)))
        #     reprojected_bboxes_2d[idx] = reprojected_bboxes_2d[idx][valid_inds]
        #     bbox3d_results[idx] = bbox3d_results[idx][valid_inds]

        debug_outputs = {
            k: v.detach().cpu().numpy()
            for k, v in debug_outputs.items()
        }

        return bbox2d_results, bbox3d_results, reprojected_bboxes_2d, debug_outputs

    def aug_test(self,
                 left_imgs,
                 right_imgs,
                 img_metas,
                 rescale=False,
                 **cam_params):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # TODO(xyg): test
        raise NotImplementedError
        # imgs = left_imgs
        # # recompute feats to save memory
        # proposal_list = self.aug_test_rpn(
        #     self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        # det_bboxes, det_labels = self.aug_test_bboxes(
        #     self.extract_feats(imgs), img_metas, proposal_list,
        #     self.test_cfg.rcnn)

        # if rescale:
        #     _det_bboxes = det_bboxes
        # else:
        #     _det_bboxes = det_bboxes.clone()
        #     _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        # bbox_results = bbox2result(_det_bboxes, det_labels,
        #                            self.bbox_head.num_classes)

        # # det_bboxes always keep the original scale
        # if self.with_mask:
        #     segm_results = self.aug_test_mask(
        #         self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
        #     return bbox_results, segm_results
        # else:
        #     return bbox_results

    def show_result(self,
                    data,
                    result,
                    dataset=None,
                    score_thr=0.,
                    show=True,
                    out_file=None,
                    annos=None,
                    K=None):
        bbox2d_result, bbox3d_result, reproj_result, debug_results = result

        img_tensor = data['left_img'][0].data[0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        class_names = tuple([name[:3] for name in class_names])

        def format_labels(bbox_result):
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            return np.concatenate(labels)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            h_ori, w_ori, _ = img_meta['ori_shape']
            img_show = img[:h, :w, :]
            img_show = mmcv.imresize(img_show, (w_ori, h_ori))

            img_show_2d = img_show.copy()
            img_show_3d = img_show.copy()
            img_show_reproj = img_show.copy()
            img_show_bev = np.ones([800, 800, 3], np.uint8) * 255
            del img_show

            # draw 2d detection results
            bboxes2d = np.vstack(bbox2d_result)
            labels2d = format_labels(bbox2d_result)
            if annos is not None:
                mmcv.imshow_det_bboxes(
                    img_show_2d,
                    annos['bboxes'],
                    annos['labels'] - 1,
                    class_names=class_names,
                    bbox_color='green',
                    text_color='green',
                    show=False)
            mmcv.imshow_det_bboxes(
                img_show_2d,
                bboxes2d,
                labels2d,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                bbox_color='red',
                text_color='red',
                out_file=None)

            # draw 2d reproj results
            bboxes_reproj = np.vstack(reproj_result)
            labels_reproj = format_labels(reproj_result)
            if annos is not None:
                mmcv.imshow_det_bboxes(
                    img_show_reproj,
                    annos['bboxes'],
                    annos['labels'] - 1,
                    class_names=class_names,
                    bbox_color='green',
                    text_color='green',
                    show=False)
            mmcv.imshow_det_bboxes(
                img_show_reproj,
                bboxes_reproj,
                labels_reproj,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                bbox_color='red',
                text_color='red',
                out_file=None)

            # # draw 3d results
            assert K is not None
            K = K[0].data[0][0].detach().cpu().numpy().copy()
            K[:2] /= img_meta['scale_factor']

            bboxes3d = np.vstack(bbox3d_result)
            labels3d = format_labels(bbox3d_result)
            if annos is not None:
                imshow_3d_det_bboxes(
                    img_show_3d,
                    get_corners_from_3d_bboxes(annos['bboxes_3d'], K),
                    annos['labels'] - 1,
                    class_names=class_names,
                    bbox_color='green',
                    text_color='green',
                    show=False)
            imshow_3d_det_bboxes(
                img_show_3d,
                get_corners_from_3d_bboxes(bboxes3d, K),
                labels3d,
                scores=bboxes3d[:, 7],
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                bbox_color='red',
                text_color='red',
                out_file=None,
                random_color=True)

            if annos is not None:
                imshow_bev_bboxes(
                    img_show_bev, (-40, 0, 40, 80.),
                    get_bev_from_3d_bboxes(annos['bboxes_3d']),
                    annos['labels'] - 1,
                    class_names=class_names,
                    bbox_color='green',
                    text_color='green',
                    show=False)
            imshow_bev_bboxes(
                img_show_bev, (-40, 0, 40, 80.),
                get_bev_from_3d_bboxes(bboxes3d),
                labels3d,
                scores=bboxes3d[:, 7],
                class_names=class_names,
                score_thr=score_thr,
                bbox_color='red',
                text_color='red',
                show=False,
                out_file=None)
            img_show_bev2 = img_show_bev.copy()
            if 'roi_points_preds' in debug_results:
                imshow_bev_points(
                    img_show_bev, (-40, 0, 40, 80.),
                    get_bev_from_3d_points(debug_results['roi_points_preds']),
                    point_color='black',
                    thickness=1,
                    show=False)
            if annos is not None and 'roi_points_gts' in debug_results:
                imshow_bev_points(
                    img_show_bev2, (-40, 0, 40, 80.),
                    get_bev_from_3d_points(debug_results['roi_points_gts']),
                    point_color='blue',
                    thickness=1,
                    show=False)

            img_combine = np.concatenate(
                [img_show_2d, img_show_3d, img_show_reproj], 0)
            img_combine = np.concatenate([
                img_combine,
                np.pad(img_show_bev,
                       [(0, img_combine.shape[0] - img_show_bev.shape[0]),
                        (0, 0), (0, 0)]),
                np.pad(img_show_bev2,
                       [(0, img_combine.shape[0] - img_show_bev2.shape[0]),
                        (0, 0), (0, 0)])
            ], 1)
            mmcv.imwrite(img_combine,
                         out_file.rsplit(".", 1)[0] + "_combine.jpg")


def get_corners_from_3d_bboxes(bboxes_3d, K):
    corners = box_np_ops.center_to_corner_box3d(
        bboxes_3d[:, :3],
        bboxes_3d[:, 3:6],
        bboxes_3d[:, 6],
        origin=[0.5, 1.0, 0.5],
        axis=1)
    corners_2d = corners @ K.T
    corners_2d = corners_2d[..., :2] / corners_2d[..., 2:3]
    return corners_2d


def get_bev_from_3d_bboxes(bboxes_3d):
    corners = box_np_ops.center_to_corner_box3d(
        bboxes_3d[:, :3],
        bboxes_3d[:, 3:6],
        bboxes_3d[:, 6],
        origin=[0.5, 1.0, 0.5],
        axis=1)
    return corners[:, [0, 1, 5, 4]][:, :, [0, 2]]


def get_bev_from_3d_points(points_3d):
    assert len(points_3d.shape) == 2
    points_3d = points_3d[~np.isnan(points_3d.mean(1))]
    return points_3d[:, [0, 2]]


def imshow_3d_det_bboxes(img,
                         corners_or_3d_bboxes,
                         labels,
                         scores=None,
                         class_names=None,
                         score_thr=0,
                         bbox_color='green',
                         text_color='green',
                         thickness=1,
                         font_scale=0.3,
                         show=True,
                         win_name='',
                         wait_time=0,
                         out_file=None,
                         random_color=False,
                         K=None):
    """Draw 3d bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        corners (ndarray): Bounding boxes (with scores), shaped (n, 8, 2).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    if len(corners_or_3d_bboxes.shape) == 3:
        corners = corners_or_3d_bboxes
    else:
        corners = get_corners_from_3d_bboxes(
            corners_or_3d_bboxes, K, return_z=False)
    assert corners.ndim == 3
    assert labels.ndim == 1
    assert corners.shape[0] == labels.shape[0]
    assert corners.shape[1] == 8
    img = mmcv.imread(img)

    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == labels.shape[0]

    bbox_color = mmcv.color_val(bbox_color)
    text_color = mmcv.color_val(text_color)

    for idx, (corner, label) in enumerate(zip(corners, labels)):
        corner = np.round(corner).astype(np.int32)
        if random_color:
            bbox_color = (list(np.random.choice(range(256), size=3)))
            bbox_color = [255, int(bbox_color[1]), int(bbox_color[2])]
        for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                       (7, 4), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(
                img,
                tuple(corner[i1]),
                tuple(corner[i2]),
                bbox_color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if scores is not None:
            label_text += "|" + str(scores[idx])
        cv2.putText(img, label_text, (corner[0, 0], corner[0, 1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)


def imshow_bev_bboxes(img,
                      space_range,
                      corners,
                      labels,
                      scores=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.3,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw 3d bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        corners (ndarray): bev bounding boxes, shaped (n, 4, 2).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert corners.ndim == 3
    assert labels.ndim == 1
    assert corners.shape[0] == labels.shape[0]
    assert corners.shape[1] == 4
    # img = mmcv.imread(img)
    assert isinstance(space_range, tuple) and len(space_range) == 4

    img_size = img.shape[:2]

    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == labels.shape[0]

    # corners x [-40, 40] z [0, 80]
    # after normalize x[-0.5, 0.5] z [0, 1]
    corners = corners / \
        (np.array(space_range[2:]) -
         np.array(space_range[: 2])) * np.array(img_size)
    corners[:, :, 0] += img_size[0] / 2.
    corners[:, :, 1] = img_size[1] - corners[:, :, 1]  # flip up-bottom

    bbox_color = mmcv.color_val(bbox_color)
    text_color = mmcv.color_val(text_color)

    for idx, (corner, label) in enumerate(zip(corners, labels)):
        corner = np.round(corner).astype(np.int32)
        for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(
                img,
                tuple(corner[i1]),
                tuple(corner[i2]),
                bbox_color,
                thickness=thickness * 2 if i1 == 2 else thickness,
                lineType=cv2.LINE_AA)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if scores is not None:
            label_text += "|" + str(scores[idx])
        cv2.putText(img, label_text, (corner[0, 0], corner[0, 1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)


def imshow_bev_points(img,
                      space_range,
                      points,
                      point_color='green',
                      thickness=1,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw 3d bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        corners (ndarray): bev bounding boxes, shaped (n, 4, 2).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert points.ndim == 2
    assert isinstance(space_range, tuple) and len(space_range) == 4

    img_size = img.shape[:2]

    # corners x [-40, 40] z [0, 80]
    # after normalize x[-0.5, 0.5] z [0, 1]
    points = points / \
        (np.array(space_range[2:]) -
         np.array(space_range[: 2])) * np.array(img_size)
    points[:, 0] += img_size[0] / 2.
    points[:, 1] = img_size[1] - points[:, 1]  # flip up-bottom

    point_color = mmcv.color_val(point_color)

    # for idx, (corner, label) in enumerate(zip(corners, labels)):
    #     corner = np.round(corner).astype(np.int32)
    #     for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
    #         cv2.line(
    #             img, tuple(corner[i1]), tuple(corner[i2]), bbox_color, thickness=thickness * 2 if i1 == 2 else thickness, lineType=cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(img, tuple(np.round(point).astype(np.int)), 0, point_color,
                   thickness)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
