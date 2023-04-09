from typing import List

import cv2
import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile, Resize
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData
from shapely.geometry import Polygon

from mmocr.datasets.transforms import PackTextDetInputs, Resize
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils import register_all_modules


def build_model(config_path, ckpt_path):
    """Build model from config file.
    
    Args:
        config_path (str): Path to config file.
        ckpt_path (str): Path to checkpoint file.
    """
    # build model
    model = MODELS.build(Config.fromfile(config_path).model)
    model.init_weights()
    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    _load_checkpoint_to_model(model, checkpoint)
    return model


def construct_batch(imgs):
    """Construct batch from data infos for DBNet.

    Args:
        data_infos (list[dict]): List of data infos for LayoutLM with the
            following keys:
                - img_path (str): Path to image file.
                - image_width (int): Width of image.
                - image_height (int): Height of image.
                - instances (list[dict]): List of instances with the following
                    keys:
                    - text (str): Text of instance.
                    - bbox (list[float]): Bounding box of instance.
                    - polygon (list[list[float]]): Polygon of instance.
        resize_scale (tuple[int]): Scale of image. Default: (160, 160).
    """
    data_samples = []
    for img in imgs:
        data_sample = TextDetDataSample()
        data_samples.append(data_sample)
    return dict(inputs=imgs, data_samples=data_samples)


def dbresults2cotext(db_results):
    """Convert DBNet results to CoText format.

    Args:
        db_results list[TextDetDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
    
    Returns:
        cotext_results (dict): A dict of CoText format with the following keys:
            - bboxes List(ndarray): shape np.array([8], dtype=int32)
            - scores List(float): detection score
            - instances

    """
    cotext_results = []
    for db_result in db_results:
        cotext_result = dict(
            text=db_result['text'],
            bbox=db_result['bbox'],
            polygon=db_result['polygon'])
        cotext_results.append(cotext_result)
    return cotext_results


def pack_det_results(results, img_shape):
    """Pack detection results to InstanceData.

    Args:
        results (list[dict]): List of detection results with the following keys:
            - bboxes (list[list[float]]): Bounding boxes of instances.
            - scores (list[float]): Classification scores of instances.
            - polygons (list[list[list[float]]]): Polygons of instances.
        img_shape (tuple[int]): Shape of image.
    """
    packed_results = []

    for result in results:
        polygons = result.pred_instances.polygons
        scores = result.pred_instances.scores.cpu().numpy().tolist()
        areas = []
        bboxes = [[0, 0, 0, 0]]
        instances = []
        # create a np mask as the shape of image
        mask = np.zeros(img_shape, dtype=np.float32)
        for i, polygon in enumerate(polygons):
            current_idx = i + 1
            # convert polygon to shapely and compute area
            polygon = np.array(polygon)
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            # assign current_idx to the mask
            cv2.fillPoly(mask, [polygon], current_idx)
            # calculate bbox
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            bboxes.append([x_min, x_max, y_min, y_max])
            polygon = polygon.tolist()
            polygon = Polygon(polygon)
            areas.append(polygon.area)
            instances.append(current_idx)
        if len(instances) == 0:
            instances = [0]
        packed_results.append(
            dict(
                bboxes=polygons,
                scores=scores,
                areas=areas,
                label=mask,
                bboxes_h=np.array([bboxes]).astype(np.float32),
                instances=[instances]))
    return packed_results


def db_forward(imgs: List[str], model: torch.nn.Module, mode: str = 'test'):
    """Forward DBNet++ to get features and detection results.

    Args:
        imgs (torch.Tensor): Tensor of shape (N, 3, H, W).
        model (torch.nn.Module): DBNet++ model.
        mode (str): 'test' or 'train'. Default: 'test'.
    """

    batch = construct_batch(imgs)
    model.eval()
    with torch.no_grad():
        inputs, data_sample = batch['inputs'], batch['data_samples']
        if mode == 'test':
            features, det_results = model.predict(
                inputs, data_sample, mode='test')
            return features, pack_det_results(det_results, imgs.shape[2:])
        elif mode == 'train':
            features = model.predict(inputs, None, mode='train')
            return features


if __name__ == '__main__':
    config_path = 'configs/textdet/dbnetpp/dstext/_base_dbnetpp_resnet50-dcnv2_fpnc.py'
    ckpt_path = 'work_dirs/dbnetpp_resnet50-dcnv2_fpnc_100e_dstext/hmean_4754/epoch_100.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path)
    imgs = torch.randn(2, 3, 640, 640)
    img_metas = None  # used for test
    features = db_forward(imgs, model, mode='train')
    print(features.shape)
    features, det_results = db_forward(imgs, model, mode='test')
    print(det_results)