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


def construct_batch(data_infos):
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
    """
    loading = LoadImageFromFile(color_type='color_ignore_orientation')
    resize = Resize(scale=(160, 160), keep_ratio=True)
    packing = PackTextDetInputs(
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    results = dict(inputs=[], data_samples=[])
    for data_info in data_infos:
        result = dict(img_path=data_info['img_path'])
        result = loading(result)
        result['gt_bboxes'] = np.array(
            [instance['bbox'] for instance in data_info['instances']])
        result = resize(result)
        result = packing(result)
        results['inputs'].append(result['inputs'])
        results['data_samples'].append(result['data_samples'])
    return results


def dbnetpp_roi_forward_batch(model, batch):
    """Forward batch of images for DBNet++ with RoIAlign.

    Args:
        model (nn.Module): DBNet++ with RoIAlign.
        batch (dict): Batch of images.

    Returns:
        roi_samples (list[Dict]): list of roi features. The length of the list
            equals to the batch size. Each element is a dict with the following
            keys:
                - img_path (str): Name of image.
                - roi_features (list[dict]): List of roi features with the
                    following keys:
                    - bbox (list[float]): Bounding box of roi.
                    - roi_feature (torch.Tensor): Feature of roi. The dimension
                        should be C, C is the number of channels (Equal to 256)
    """
    model.eval()
    with torch.no_grad():
        batch = model.data_preprocessor(batch, training=False)
        inputs, data_sample = batch['inputs'], batch['data_samples']
        roi_samples = model.predict(
            inputs, data_sample, get_roi_features=True, roi_stride=4)
    return roi_samples


if __name__ == '__main__':
    config_path = 'configs/textdet/dbnetpp/dstext/_base_dbnetpp_resnet50-dcnv2_fpnc.py'
    ckpt_path = 'work_dirs/dbnetpp_resnet50-dcnv2_fpnc_100e_dstext/hmean_4754/epoch_100.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path)
    # batch size equals to 2
    data_infos = [
        dict(
            img_path=
            '/media/jiangqing/jqssd/ICDAR-2023/data/hiertext/det_images/val/0a2c4ca4b143b784.jpg',
            image_width=4068,
            image_height=1024,
            instances=[
                dict(
                    text='1',
                    bbox=[0.0, 0.0, 20.0, 20.0],
                    polygon=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                dict(
                    text='2',
                    bbox=[10.0, 12.0, 30.0, 30.0],
                    polygon=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            ]),
        dict(
            img_path=
            '/media/jiangqing/jqssd/ICDAR-2023/data/hiertext/det_images/val/0a85661607f14abf.jpg',
            image_width=4068,
            image_height=1024,
            instances=[
                dict(
                    text='3',
                    bbox=[0.0, 0.0, 20.0, 20.0],
                    polygon=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                dict(
                    text='4',
                    bbox=[1.0, 2.0, 20.0, 20.0],
                    polygon=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            ])
    ]
    batch = construct_batch(data_infos)
    roi_samples = dbnetpp_roi_forward_batch(model, batch)
    for roi_sample in roi_samples:
        print(roi_sample['img_path'])
        for roi_feature in roi_sample['roi_features']:
            print(roi_feature['bbox'])
            print(roi_feature['roi_feature'].shape)
