import os
import os.path as osp
import pickle

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

def visualize_det(det_path: str, out_path):
    """Visualize detection results and save to video.
    """
    with open(det_path, 'rb') as f:
        det_results = pickle.load(f)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # sort det_results by name
    det_results = sorted(det_results, key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))
    imgs = []
    for i, det_result in tqdm(enumerate(det_results)):
        img_path = det_result['img_path']
        polygons = det_result['pred_instances']['polygons']
        img = cv2.imread(img_path)
        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1,2).astype(np.int32)
            cv2.polylines(img, [polygon], True, (0, 255, 0), 3)
        imgs.append(img)
        cv2.imwrite(os.path.join(out_path, f'frame{i}.jpg'), img)
    # save to video
    mmcv.frames2video(
        out_path,
        osp.join(out_path, 'vis.mp4'),
        filename_tmpl='frame{:d}.jpg',
        fps=20,
        fourcc='mp4v')

if __name__ == '__main__':
    visualize_det(
        'dstext_test/Video_162_6_2/det_out/epoch_60.pth_predictions.pkl',
        'dstext_test/Video_162_6_2/det_vis/')
