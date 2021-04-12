import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner


@BBOX_ASSIGNERS.register_module()
class ATSS3DCenterAssigner(ATSSAssigner):
    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 append_3d_centers=True,
                 thresh_mode='meanstd',
                 thresh_ratio=0.75):
        super(ATSS3DCenterAssigner, self).__init__(topk=topk,
                                                   iou_calculator=iou_calculator,
                                                   ignore_iof_thr=ignore_iof_thr)
        self.append_3d_centers = append_3d_centers
        self.thresh_mode = thresh_mode
        assert self.thresh_mode in ['meanstd', 'ratio']
        self.thresh_ratio = thresh_ratio

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        if self.append_3d_centers:
            gt_bboxes = gt_bboxes[:, :6]
            assert gt_bboxes.shape[1] == 6
        else:
            gt_bboxes = gt_bboxes[:, :4]
            assert gt_bboxes.shape[1] == 4
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes[:, :4])

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        if not self.append_3d_centers:
            gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
            gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        else:
            # use self-defined gt boxes center
            gt_points = gt_bboxes[:, 4:6]

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt centeroov
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        if self.thresh_mode == 'meanstd':
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        elif self.thresh_mode == 'ratio':
            thresh_ratio = self.thresh_ratio
            candidate_overlaps_by_lvl = candidate_overlaps.view(len(num_level_bboxes), selectable_k, num_gt)
            overlaps_mean_by_lvl = candidate_overlaps_by_lvl.mean(1, keepdim=True)
            overlaps_thr_by_lvl = overlaps_mean_by_lvl.max(0, keepdim=True).values * thresh_ratio
            is_pos_by_lvl = overlaps_mean_by_lvl > overlaps_thr_by_lvl
            is_pos = (is_pos_by_lvl.repeat(1, selectable_k, 1)).view(candidate_overlaps.shape)
        else:
            raise ValueError('invalid thresh mode')

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
