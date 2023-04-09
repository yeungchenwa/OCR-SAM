import argparse
import os

import cv2
import mmengine
import numpy as np
from tqdm import tqdm

# Create a color map, which is used to visualize the text instances
MAX_INSTANCES = 2000
colors = np.random.randint(0, 255, (MAX_INSTANCES, 3))
COLORMAP = {}
for i in range(MAX_INSTANCES):
    COLORMAP[i] = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('img_root', default='/home/jq/ICDAR-2023-HierText/det_images/val', help='the root of the images')
    parser.add_argument('anno_path', default='/home/jq/projects/MAE/ICDAR-2023-line_detection/ICDAR-2023/mmocr-dev-1.x@9b0f1da/results/hiertext-baseline-20230305/LayoutAnalysis_mmocr1.0.json', help='the path of the annotation file')
    parser.add_argument(
        '--group_level',
        default='line',
        type=str,
        help='the level of the group. It can be "line" or "para"')
    parser.add_argument(
        '--output-dir',
        '-o',
        default='/home/jq/projects/MAE/ICDAR-2023-line_detection/ICDAR-2023/mmocr-dev-1.x@9b0f1da/results/hiertext-baseline-20230305/vis',
        type=str,
        help='If there is no display interface, you can save it.')
    args = parser.parse_args()
    return args


def visualize(img_root: str, anno: dict, out_path: str, group_level: str):
    """Visualize the annotation of the dataset.
    
    Args:
        img_root (str): The root of the images.
        anno_path (str): The path of the annotation file.
        out_path (str): The path of the output directory.
        group_level (str): The level of the group. It can be 'line' or 'para'.
    """
    img_path = anno['img_path'] + '.jpg'
    # img_path = anno['img_path']
    # img_name = img_path.split('/')[-1]
    instances = anno['instances']
    # load image
    img = cv2.imread(os.path.join(img_root, img_path))[:, :, ::-1]
    for instance in instances:
        text = instance['text']
        polygon = instance['polygon']
        polygon = np.array(polygon).reshape(-1, 2)
        para_id = instance['para_id']
        line_id = instance['line_id']
        # draw a mask given the polygon
        if group_level == 'para':
            color = COLORMAP[para_id]
        elif group_level == 'line':
            color = COLORMAP[line_id]
        mask = cv2.fillPoly(img.copy() * 0, [polygon.astype('int32')], color)
        # paste the mask to the image
        img = cv2.addWeighted(img, 1, mask, 0.5, 0)
    # save image
    cv2.imwrite(os.path.join(out_path, img_path), img[:, :, ::-1])


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_obj = mmengine.load(args.anno_path)
    annotations = json_obj['data_list']
    for annotation in tqdm(annotations):
        visualize(args.img_root, annotation, args.output_dir, args.group_level)