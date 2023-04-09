import json
import os

import cv2
import numpy as np


def visualize(img_dir: str,
              jsonl_path: str,
              vis_dir: str,
              vis_nums: int = 100):
    """Visualize bounding boxes in images

    Args:
        img_dir (str): Path to image folder
        jsonl_path (str): Path to jsonl file
        vis_dir (str): Path to save visualization images
        vis_nums (int, optional): Number of images to visualize. Defaults to 100.
    """
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('loading annotations...')
    with open(jsonl_path, 'r') as f:
        annos = [json.loads(line) for line in f.readlines()]
    print('visualizing...')
    for anno in annos[3000:]:
        img_path = os.path.join(img_dir, anno['img_path'])
        img = cv2.imread(img_path)
        # calculate the area of the image
        img_area = img.shape[0] * img.shape[1]
        for i, polygon in enumerate(anno['polygons']):
            polygon = np.array(polygon).reshape(-1, 1, 2).astype(np.int32)
            # calulate the area of the polygon
            text = anno['text'][i]
            # calculate the ratio of the polygon area to the image area
            cv2.polylines(img, [polygon], True, (0, 255, 0), 3)
            # display text
            cv2.putText(img, text,
                        (polygon[0][0][0], polygon[0][0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        img_name = anno['img_path'].split('/')[-1]
        cv2.imwrite(os.path.join(vis_dir, img_name), img)
        vis_nums -= 1
        if vis_nums == 0:
            break


if __name__ == '__main__':
    img_dir = '/media/jiangqing/jqnas/projects/TextCLIP/data'
    jsonl_path = '/media/jiangqing/jqssd/projects/research/TextCLIP/mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00000/e2e.jsonl'
    vis_dir = 'vis'
    visualize(img_dir, jsonl_path, vis_dir)