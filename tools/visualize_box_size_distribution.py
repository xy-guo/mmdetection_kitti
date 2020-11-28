import numpy as np
import mmcv
import argparse
import matplotlib.pyplot as plt


def plot_loghist(ax, x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x, bins=logbins)
    ax.set_xscale('log')


def get_args():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--meta_file', required=False, type=str,
                        default='data/kitti/kitti_infos_train.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    metas = mmcv.load(args.meta_file)

    for cls_name in ['Car', 'Pedestrian', "Cyclist"]:
        bboxes = [meta['annos']['bbox'][meta['annos']['name'] == cls_name]
                  for meta in metas]
        bboxes = np.concatenate(bboxes)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        area = w * h
        ratio = w / h

        print(f"for class {cls_name}, totally {len(bboxes)} boxes")

        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        plot_loghist(ax1, w, 100)
        ax1.set_title('w')
        plot_loghist(ax2, h, 100)
        ax2.set_title('h')
        plot_loghist(ax3, area, 100)
        ax3.set_title('area')
        plot_loghist(ax4, ratio, 100)
        ax4.set_title('ratio')
        plt.show()
