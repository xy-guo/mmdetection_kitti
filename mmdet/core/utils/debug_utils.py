import os.path as osp
import torch
import mmcv
import cv2
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import master_only
from mmdet.core.utils import tensor2imgs
from mmdet.utils.det3d import box_np_ops
import numpy as np


def imshow_3d_det_bboxes(img,
                         corners,
                         labels,
                         scores=None,
                         class_names=None,
                         score_thr=0,
                         bbox_color='green',
                         text_color='green',
                         thickness=1,
                         font_scale=0.5,
                         show=True,
                         win_name='',
                         wait_time=0,
                         out_file=None):
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

    for corner, label in zip(corners, labels):
        corner = np.round(corner).astype(np.int32)
        bbox_color = (list(np.random.choice(range(256), size=3)))
        bbox_color = [int(bbox_color[0]), int(
            bbox_color[1]), int(bbox_color[2])]
        for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 0),
                       (4, 5), (5, 6), (6, 7), (7, 4), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(
                img, tuple(corner[i1]), tuple(corner[i2]), bbox_color, thickness=thickness, lineType=cv2.LINE_AA)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        cv2.putText(img, label_text, (corner[0, 0], corner[0, 1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)


@HOOKS.register_module
class DebugLoggerHook(Hook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 enable=False):
        super(DebugLoggerHook, self).__init__()
        self.log_dir = log_dir
        self.enable = enable

    @master_only
    def after_train_iter(self, runner):
        if not self.enable:
            return
        # draw images
        data = runner.data_batch
        data = {k: v.data[0] for k, v in data.items()}  # data of GPU:0
        # data = {k: v[0] for k, v in data.items()}  # data of sample 0

        # available keys:
        # K, fL, gt_bboxes, gt_bboxes_ignore, gt_labels
        # img_meta, left_img, right_img, t2to3, velo2cam2, velo2cam3
        iter_idx = runner._iter
        img_metas = data['img_meta']
        left_img_tensor = data['left_img']
        right_img_tensor = data['right_img']
        gt_bboxes = data['gt_bboxes']
        gt_bboxes_3d = data['gt_bboxes_3d']
        intrinsics = data['K']
        gt_bboxes_ignore = data['gt_bboxes_ignore']

        left_imgs = tensor2imgs(
            left_img_tensor, **img_metas[0]['img_norm_cfg'])
        right_imgs = tensor2imgs(
            right_img_tensor, **img_metas[0]['img_norm_cfg'])
        mix_imgs = [(l * 0.65 + r * 0.35)
                    for l, r in zip(left_imgs, right_imgs)]

        for idx in range(len(left_imgs)):
            img_show = mix_imgs[idx].copy()
            img_show_3d = mix_imgs[idx].copy()
            bboxes = gt_bboxes[idx].detach().cpu().numpy()
            bboxes_3d = gt_bboxes_3d[idx].detach().cpu().numpy()
            K = intrinsics[idx].detach().cpu().numpy()
            corners = box_np_ops.center_to_corner_box3d(
                bboxes_3d[:, :3], bboxes_3d[:, 3:6], bboxes_3d[:, 6], origin=[0.5, 1.0, 0.5], axis=1)
            bboxes_ignore = gt_bboxes_ignore[idx].detach().cpu().numpy()

            labels = data['gt_labels'][idx].detach().cpu().numpy()
            labels_ignore = np.array([0] * len(bboxes_ignore))
            swap = img_metas[idx]['swap']
            flip = img_metas[idx]['flip']
            filename = img_metas[idx]['left_filename']

            cv2.putText(img_show, "swap " + str(swap), (10, 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img_show, "flip " + str(flip), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img_show, filename, (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img_show_3d, "swap " + str(swap), (10, 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img_show_3d, "flip " + str(flip), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img_show_3d, filename, (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=['?', 'c', 'p', 'b'],
                bbox_color='green',
                score_thr=0.,
                show=False)

            corners_2d = corners @ K.T
            corners_2d = corners_2d[..., :2] / corners_2d[..., 2:3]
            imshow_3d_det_bboxes(
                img_show_3d,
                corners_2d,
                labels,
                class_names=['?', 'c', 'p', 'b'],
                bbox_color='green',
                score_thr=0.,
                show=False,
                out_file=osp.join(self.log_dir, f'debug_{iter_idx:06d}_{idx:02d}_3d.jpg'))
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes_ignore,
                labels_ignore,
                class_names=['x'],
                bbox_color='red',
                score_thr=0.,
                show=False,
                out_file=osp.join(self.log_dir, f'debug_{iter_idx:06d}_{idx:02d}.jpg'))
            print("saving debug img to ", self.log_dir,
                  iter_idx, idx, "swap", swap, filename)
