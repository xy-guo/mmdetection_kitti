#!/usr/bin/env bash
./dist_train.sh ./configs/kitti/faster_rcnn_r50_fpn_1x_kitti.py 8 -seed 0
./dist_test.sh ./configs/kitti/faster_rcnn_r50_fpn_1x_kitti.py CHECKPOINT 8 