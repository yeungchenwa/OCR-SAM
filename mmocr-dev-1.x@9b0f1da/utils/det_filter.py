import concurrent.futures
import json
import os
import time

import cv2
import imagesize
import numpy as np
from tqdm import tqdm


def filter_anno(anno, img_dir, min_score, min_area_ratio):
    img_path = os.path.join(img_dir, anno['img_path'])
    width, height = imagesize.get(img_path)
    img_shape = (height, width)
    img_area = int(img_shape[0]) * int(img_shape[1])
    polygons = []
    scores = []
    for i, polygon in enumerate(anno['polygons']):
        polygon = np.array(polygon).reshape(-1, 1, 2).astype(np.int32)
        poly_area = cv2.contourArea(polygon)
        ratio = poly_area / img_area
        score = anno['scores'][i]
        if score <= min_score:
            continue
        if ratio <= min_area_ratio:
            continue
        polygons.append(polygon.tolist())
        scores.append(score)
    return {
        'img_path': anno['img_path'],
        'polygons': polygons,
        'scores': scores
    }


def det_filter(img_dir, det_results, out_path, min_score, min_area_ratio):
    """Filter detection results

    Args:
        img_dir (str): Path to image folder
        det_results (list): Detection results
        out_path (str): Path to save filtered detection results
        min_score (float): Minimum score of detection results
        min_area_ratio (float): Minimum area ratio of detection results
    """
    with open(det_results, 'r') as f:
        annos = [json.loads(line) for line in f.readlines()]
    new_annos = []
    ori_num_instances = 0
    new_num_instances = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_anno = {}
        for anno in annos:
            future = executor.submit(filter_anno, anno, img_dir, min_score,
                                     min_area_ratio)
            future_to_anno[future] = anno
        loop = tqdm(concurrent.futures.as_completed(future_to_anno))
        for future in loop:
            anno = future_to_anno[future]
            new_anno = future.result()
            new_annos.append(new_anno)
            ori_num_instances += len(anno['polygons'])
            new_num_instances += len(new_anno['polygons'])
            loop.set_description(
                f'ori_num_instances: {ori_num_instances}, new_num_instances: {new_num_instances}'
            )
    with open(out_path, 'w') as f:
        for anno in new_annos:
            f.write(json.dumps(anno))
            f.write('\n')
    print(
        f'ori_num_instances: {ori_num_instances}, new_num_instances: {new_num_instances}'
    )

def fix_det(det_results, out_path):
    """Fix the incorrect polygon format

    Args:
        det_results (list): Detection results
        out_path (str): Path to save fixed detection results
    """
    with open(det_results, 'r') as f:
        annos = [json.loads(line) for line in f.readlines()]
    new_annos = []
    for anno in tqdm(annos):
        new_anno = {
            'img_path': anno['img_path'],
            'polygons': [],
            'scores': []
        }
        for i, polygon in enumerate(anno['polygons']):
            polygon = np.array(polygon).reshape(-1).astype(np.int32)
            new_anno['polygons'].append(polygon.tolist())
            new_anno['scores'].append(anno['scores'][i])
        new_annos.append(new_anno)
    with open(out_path, 'w') as f:
        for anno in new_annos:
            f.write(json.dumps(anno))
            f.write('\n')

if __name__ == '__main__':
    img_dir = '/media/jiangqing/jqnas/projects/TextCLIP/data'
    det_results = 'mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00001/det_out/det_results.jsonl'
    out_path = 'mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00001/det_out/det_results_filtered_ms0.6_mar_0.002.jsonl'
    min_score = 0.6
    min_area_ratio = 0.002
    det_filter(img_dir, det_results, out_path, min_score, min_area_ratio)
    # fix_det(
    #     'mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00000/det_out/det_results_part-00000_filtered_ms0.6_mar_0.002.jsonl',
    #     'mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00000/det_out/det_results_part-00000_filtered_ms0.6_mar_0.002_fix.jsonl'
    # )
