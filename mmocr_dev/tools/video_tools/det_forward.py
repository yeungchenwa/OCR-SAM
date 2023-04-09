from typing import List

import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile, Resize
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData
from mmocr.datasets.transforms import PackTextDetInputs, Resize
from mmocr.registry import MODELS
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


def construct_batch(data_infos, resize_scale=(160, 160)):
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
    loading = LoadImageFromFile(color_type='color_ignore_orientation')
    resize = Resize(scale=resize_scale, keep_ratio=True)
    packing = PackTextDetInputs(
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    results = dict(inputs=[], data_samples=[])
    for data_info in data_infos:
        result = dict(img_path=data_info['img_path'])
        result = loading(result)
        result = resize(result)
        result = packing(result)
        results['inputs'].append(result['inputs'])
        results['data_samples'].append(result['data_samples'])
    return results


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


def db_forward(img_paths: List[str],
               model: torch.nn.Module,
               mode: str = 'test'):
    """Forward DBNet++ to get features and detection results.

    Args:
        img_paths (list[str]): List of image paths.
        config_path (str): Path to config file.
        ckpt_path (str): Path to checkpoint file.
    """
    data_infos = []
    for img_path in img_paths:
        data_infos.append(dict(img_path=img_path))
    batch = construct_batch(data_infos)
    model.eval()
    with torch.no_grad():
        batch = model.data_preprocessor(batch, training=False)
        inputs, data_sample = batch['inputs'], batch['data_samples']
        if mode == 'test':
            features, det_results = model.predict(
                inputs, data_sample, mode='test')
            return features, det_results
        elif mode == 'train':
            features = model.predict(inputs, data_sample, mode='train')
            return features


if __name__ == '__main__':
    config_path = 'configs/textdet/dbnetpp/dstext/_base_dbnetpp_resnet50-dcnv2_fpnc.py'
    ckpt_path = 'work_dirs/dbnetpp_resnet50-dcnv2_fpnc_100e_dstext/hmean_4754/epoch_100.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path)
    img_paths = [
        '/media/jiangqing/jqssd/ICDAR-2023/data/hiertext/det_images/val/0a2c4ca4b143b784.jpg',
        '/media/jiangqing/jqssd/ICDAR-2023/data/hiertext/det_images/val/0a85661607f14abf.jpg'
    ]
    features = db_forward(img_paths, model, mode='train')
    print(features.shape)
    features, det_results = db_forward(img_paths, model, mode='test')
    print(det_results)