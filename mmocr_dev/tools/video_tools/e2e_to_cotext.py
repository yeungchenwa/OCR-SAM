import json
import pickle
import os
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

from mmocr.utils.polygon_utils import poly2bbox


def e2e2cotext(e2e_path: str, out_path: str, scale_shape=(800, 1424)):
    '''Convert e2e results to cotext format.

    Args:
        e2e_path (str): Path to e2e results.
        out_path (str): Path to cotext results.
        scale_shape (tuple(int, int)): The height and width of the scaled
            image. Default: (800, 1432).
    '''
    with open(e2e_path, 'rb') as f:
        e2e_results = pickle.load(f)
    cotext_results = []
    for e2e_result in e2e_results:
        img_shape = e2e_result['ori_shape']  # (h, w)
        img_path = '/'.join(e2e_result['img_path'].split('/')[-2:])
        pred_instances = e2e_result['pred_instances']
        polygons = pred_instances['polygons']
        # convert polygons to a dict: 0: [], 1: [], 2: [], 3: []
        polygons_dict = {}
        for i, polygon in enumerate(polygons):
            # scale the polygon to scale_shape
            polygon = np.array(polygon).reshape(-1, 2)
            polygon[:, 0] = polygon[:, 0] * scale_shape[1] / img_shape[1]
            polygon[:, 1] = polygon[:, 1] * scale_shape[0] / img_shape[0]
            polygons_dict[i] = polygon.astype(np.int32).flatten().tolist()
        bboxes = [poly2bbox(polygon) for polygon in polygons]
        # convert bboxes to a dict: 0: [], 1: [], 2: [], 3: []
        bboxes_dict = {}
        for i, bbox in enumerate(bboxes):
            bbox = bbox.astype(np.int32).tolist()
            # scale the bbox to scale_shape
            bbox[0] = bbox[0] * scale_shape[1] / img_shape[1]
            bbox[1] = bbox[1] * scale_shape[0] / img_shape[0]
            bbox[2] = bbox[2] * scale_shape[1] / img_shape[1]
            bbox[3] = bbox[3] * scale_shape[0] / img_shape[0]
            bboxes_dict[i] = bbox
        texts = pred_instances['text']
        # convert texts to a dict: 0: '', 1: '', 2: '', 3: ''
        texts_dict = {}
        for i, text in enumerate(texts):
            texts_dict[i] = text
        cotext_result = {
            'img_shape': img_shape,
            'img_path': img_path,
            'polygons': polygons_dict,
            'bboxes': bboxes_dict,
            'texts': texts_dict
        }
        cotext_results.append(cotext_result)
    # save cotext results
    # sort cotext results by img_path
    cotext_results = sorted(
        cotext_results,
        key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))
    #key=lambda x: int(x['img_path'].split('frame')[-1].split('.')[0]))
    with open(out_path, 'w') as f:
        for result in cotext_results:
            f.write(json.dumps(result, ensure_ascii=False))
            f.write('\n')


def load_cotext(e2e_path: str, img_shape=(800, 1424)):
    """Load cotext results from a jsonl file

    Args:
        e2e_path (str): Path to cotext results.
    
    """
    packed_results = []
    with open(e2e_path, 'r') as f:
        cotext_results = [json.loads(line) for line in f.readlines()]
    for cotext_result in cotext_results:
        ori_shape = cotext_result['img_shape']
        img_name = cotext_result[
            'img_path']  #Activity_Video_163_6_3/frame1.jpg
        areas = []
        bboxes = [[0, 0, 0, 0]]
        instances = []
        polygons = cotext_result['polygons']
        texts = cotext_result['texts']
        mask = np.zeros(img_shape, dtype=np.float32)
        new_polygons = []
        for i in range(len(polygons)):
            polygon = polygons[str(i)]
            current_idx = i + 1
            new_polygons.append(polygon)
            polygon = np.array(polygon)
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            # assign current_idx to the mask
            cv2.fillPoly(mask, [polygon], current_idx)
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            bboxes.append([y_min, x_min, y_max, x_max])
            polygon = polygon.tolist()
            polygon = Polygon(polygon)
            areas.append(polygon.area)
            instances.append(current_idx)
        if len(instances) == 0:
            instances = [0]
        packed_results.append(
            dict(
                ori_shape=ori_shape,
                img_name=img_name,
                texts=texts,
                bboxes=new_polygons,
                scores=[1.0] * len(polygons),
                areas=areas,
                label=mask,
                bboxes_h=np.array([bboxes]).astype(np.float32),
                instances=[instances]))
    return packed_results


if __name__ == '__main__':
    videos = os.listdir('dstext_test')
    for video in tqdm(videos):
        e2e_path = os.path.join('dstext_test', video, 'e2e.pkl')
        out_path = os.path.join('dstext_to_cotext', f'{video}_cotext.jsonl')
        e2e2cotext(e2e_path, out_path)
