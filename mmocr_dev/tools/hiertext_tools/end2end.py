import json
import os
import pickle
from typing import Dict, List

import cv2
import mmengine
import numpy as np
from mmocr.utils import crop_img, offset_polygon, poly2bbox
from tqdm import tqdm


def affine_crop(img, polygon, expand_ratio=0.1):
    """Crop the text instance from the image. The text instance is defined by
    the polygon. We use the affine transformation to crop the text instance and
    rotate it to horizontal or vertical.

    Args:
        img (ndarray): The image.
        polygon (ndarray): The polygon of the text instance.
        expand_ratio (float): The ratio to expand the polygon. Default: 0.1.
    """
    # calculate the bounding rectangle of the polygon
    polygon = np.array(polygon).reshape(-1, 2)
    poly_width = np.max(polygon[:, 0]) - np.min(polygon[:, 0])
    poly_height = np.max(polygon[:, 1]) - np.min(polygon[:, 1])
    offset_pixel = min(poly_width, poly_height) * expand_ratio
    polygon = offset_polygon(polygon, offset_pixel).reshape(-1,
                                                            2).astype(np.int32)

    # Define the bounding rectangle for the polygon
    rect = cv2.minAreaRect(polygon.astype(np.int32))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Crop the image to the bounding rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (width, height))

    # Rotate the cropped image to horizontal or vertical
    angle = rect[2]
    if angle < -45:
        angle += 90
        height, width = cropped.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        cropped = cv2.warpAffine(
            cropped, M, (width, height), flags=cv2.INTER_CUBIC)
    return cropped


def crop_text_instances(img_prefix: str,
                        det_pred: str,
                        recog_out_dir: str = None,
                        expand_ratio=0.1,
                        use_affine_crop=False):
    """Crop text instances from the detection results.

    Args:
        img_prefix (str): Prefix name for the image.
        det_pred (str): Path to the detection results. It should be a jsonl
            file.
        recog_out_dir (str): Path to the recognition output directory. If it is
            None, the cropped images will not be saved. Default: None.
        expand_ratio (float): The ratio to expand the polygon. Default: 0.1.
    """
    if not os.path.exists(recog_out_dir):
        os.makedirs(recog_out_dir)

    with open(det_pred, 'r') as f:
        det_results = [json.loads(line.strip()) for line in f.readlines()]

    img_idx = 0
    for det_result in tqdm(det_results):
        img_path = det_result['img_path']
        polygons = det_result['polygons']
        img = cv2.imread(os.path.join(img_prefix, img_path))
        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1).astype(np.int32).tolist()
            if use_affine_crop:
                croped_img = affine_crop(
                    img, polygon, expand_ratio=expand_ratio)
            else:
                croped_img = crop_img(img, polygon, 0.1, 0.1)
            if recog_out_dir is not None:
                try:
                    cv2.imwrite(f'{recog_out_dir}/{img_idx}.jpg', croped_img)
                    img_idx += 1
                except:
                    # write an empty image
                    cv2.imwrite(f'{recog_out_dir}/{img_idx}.jpg', img)
                    img_idx += 1
                    print('broken image')


def visualize_det(det_pred, vis_out):
    """Visualize the detection results.

    Args:
        det_pred (str): Path to the detection results. It should be a pickle
            file.
        vis_out (str): Path to the visualization output directory.
    """
    if not os.path.exists(vis_out):
        os.makedirs(vis_out)

    with open(det_pred, 'rb') as f:
        det_results = pickle.load(f)

    for det_result in tqdm(det_results):
        img_path = det_result['img_path']
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        polygons = det_result['pred_instances']['polygons']
        for poly_idx, polygon in enumerate(polygons):
            polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [polygon], True, (0, 255, 0), 2)
        cv2.imwrite(f'{vis_out}/{img_name}', img)


def construct_pseudo_recog_annos(recog_out_dir: str, out_anno_name: str):
    """Construct the pseudo annotations for the cropped text instances so that
    they can be use for test.

    Args:
        recog_out_dir (str): Path to the recognition output directory.
        out_anno_name (str): Path to the annotation output directory.
    """

    img_paths = os.listdir(recog_out_dir)
    annotation = {}
    annotation['metainfo'] = dict(
        dataset_type="TextRecogDataset", task_name="textrecog")
    data_list = []
    for img_path in tqdm(img_paths):
        img_path = os.path.join(recog_out_dir, img_path)
        text = 'bbb'
        instance = dict(text=text)
        data_list.append(dict(img_path=img_path, instances=[instance]))
    annotation['data_list'] = data_list
    mmengine.dump(annotation, out_anno_name)


def construct_pseudo_det_annos(det_img_root: str, out_anno_name: str):
    """Construct pseduo annotation for detection

    Args:
        det_img_root (str): Path to the detection image root directory.
        out_anno_name (str): Path to the annotation output directory.
    """
    img_paths = os.listdir(det_img_root)
    annotation = {}
    annotation['metainfo'] = dict(
        dataset_type="TextDetDataset",
        task_name="textdet",
        category=[dict(id=0, name='text')])
    data_list = []
    for img_path in tqdm(img_paths):
        if not img_path.endswith('.jpg'):
            continue
        instances = [
            dict(
                polygon=[0, 0, 0, 10, 10, 20, 20, 0],
                bbox=[0, 0, 10, 20],
                bbox_label=0,
                ignore=False)
        ]
        data_list.append(
            dict(
                img_path=img_path, height=800, width=800, instances=instances))
    annotation['data_list'] = data_list
    mmengine.dump(annotation, out_anno_name)


def sort_det_preds(ori_img_path: str, det_preds: List[Dict]):
    """Sort the detection results according to the order of the original image.

    Args:
        ori_img_path (str): Path to the original image.
        det_preds (List[Dict]): The detection results.
    """
    ori_img_list = os.listdir(ori_img_path)
    ori_img_list.sort()
    pass


def merge_det_recog_results(det_pred: str,
                            recog_pred: str,
                            out_pred: str = None):
    """Merge the detection and recognition results.

    Args:
        det_pred (str): Path to the detection results. It should be a jsonl
            file.
        recog_pred (str): Path to the recognition results. It should be a jsonl
            file.
        out_pred (str): Path to the output pickle file.
    """
    with open(det_pred, 'r') as f:
        det_results = [json.loads(line) for line in f.readlines()]
    with open(recog_pred, 'r') as f:
        recog_results = [json.loads(line) for line in f.readlines()]
    # sort recog results basd on the number of the image
    recog_results.sort(
        key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))
    # run a pre-checking to make sure the number of polygons equal to the
    # number of recognition results
    num_polygons = 0
    for det_result in det_results:
        num_polygons += len(det_result['polygons'])
    assert num_polygons == len(recog_results)
    new_det_results = []
    for det_result in tqdm(det_results):
        det_result['text'] = []
        det_result['text_confidence'] = []
        current_num_text = len(det_result['polygons'])
        for i in range(current_num_text):
            det_result['text'].append(
                recog_results[i]['pred_text'])
            if len(recog_results[i]['recog_score']) == 0:
                det_result['text_confidence'].append(0)
            else:
                det_result['text_confidence'].append(
                    sum(recog_results[i]['recog_score']) /
                    len(recog_results[i]['recog_score']))
        new_det_results.append(det_result)
        recog_results = recog_results[current_num_text:]
    if out_pred is not None:
        with open(out_pred, 'w') as f:
            for det_result in new_det_results:
                f.write(json.dumps(det_result) + '\n')


def visualize_e2e(img_prefix:str, e2e_path: str, out_path: str):
    """Visualize the end-to-end results.

    Args:
        im_prefix (str): Path to the image prefix.
        e2e_path (str): Path to the end-to-end results. It should be a jsonl
            file.
        out_path (str): Path to the output directory.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(e2e_path, 'r') as f:
        e2e_results = [json.loads(line) for line in f.readlines()]
    for e2e_result in tqdm(e2e_results):
        img_path = e2e_result['img_path']
        img = cv2.imread(os.path.join(img_prefix, img_path))
        polygons = e2e_result['polygons']
        texts = e2e_result['text']
        for polygon, text in zip(polygons, texts):
            polygon = np.array(polygon).astype(np.int32).reshape(-1, 2)
            polygon_ = np.array(polygon).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [polygon_], True, (225, 0, 0), 3)
            cv2.putText(img, text, (polygon[0][0] - 5, polygon[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(f'{out_path}/{os.path.basename(img_path)}', img)


if __name__ == '__main__':
    merge_det_recog_results(
        'work_dirs/LAION400M/part-00000/det_out/det_results_part-00000_filtered_ms0.6_mar_0.002_fix.jsonl',
        'work_dirs/LAION400M/part-00000/recog_out/recog_results.jsonl',
        'work_dirs/LAION400M/part-00000/e2e.jsonl')
    # visualize_e2e('/media/jiangqing/jqnas/projects/TextCLIP/data/',
    #               'work_dirs/LAION400M/part-00000/e2e.jsonl', 'vis_e2e')
