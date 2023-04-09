import time

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from mmcv.transforms import Resize
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData

from mmocr.datasets.transforms import (LoadImageFromNDArray,
                                       PackTextRecogInputs, Resize)
from mmocr.registry import MODELS
from mmocr.utils import bbox2poly, register_all_modules


def crop_img_tensor(src_img,
                    box,
                    long_edge_pad_ratio=0.4,
                    short_edge_pad_ratio=0.2):
    """Crop text region given the bounding box which might be slightly padded.
    The bounding box is assumed to be a quadrangle and tightly bound the text
    region.

    Args:
        src_img (torch.Tensor): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): The ratio of padding to the long edge. The
            padding will be the length of the short edge * long_edge_pad_ratio.
            Defaults to 0.4.
        short_edge_pad_ratio (float): The ratio of padding to the short edge.
            The padding will be the length of the long edge *
            short_edge_pad_ratio. Defaults to 0.2.

    Returns:
        torch.Tensor: The cropped image.
    """
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0
    h, w = src_img.shape[1:]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    shorter_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * shorter_size
        vertical_pad = short_edge_pad_ratio * shorter_size
    else:
        horizontal_pad = short_edge_pad_ratio * shorter_size
        vertical_pad = long_edge_pad_ratio * shorter_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    return src_img[:, top:bottom, left:right]


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
    """Construct batch from data infos for Recognizer.
    Args:
        data_infos (list[dict]): List of data infos for Recognizer with the
            following keys:
                - img_path (str): Path to image file.
                - polygons (list[list[float]]): List of polygons of text
                    instances.
    
    Returns:
        List[dict]: List of batch data infos with the following keys:
    """
    batch_data_infos = []
    packing = PackTextRecogInputs(meta_keys=('img_shape', 'valid_ratio'))
    for data_info in data_infos:
        img = data_info['img']
        polygons = data_info['polygons']
        current_image_infos = dict(inputs=[], data_samples=[])
        for polygon in polygons:
            img_crop = crop_img_tensor(img, bbox2poly(polygon).tolist())
            results = dict(ori_shape=img_crop.shape[1:])
            # resize img_crop to (32, 128)
            img_crop = F.interpolate(
                img_crop.unsqueeze(0),
                size=(32, 128),
                mode='bilinear',
                align_corners=False).squeeze(0)
            results = dict(img=img_crop)
            results = dict(img_shape=img_crop.shape[1:])
            results = packing(results)
            current_image_infos['inputs'].append(img_crop)
            current_image_infos['data_samples'].append(results['data_samples'])
        current_image_infos['inputs'] = torch.stack(
            current_image_infos['inputs'], dim=0)
        batch_data_infos.append(current_image_infos)
    return batch_data_infos


def recog_forward_batch(model, batch):
    """Forward batch of images for Recognizer.
    Args:
        model (nn.Module): Recognizer.
        batch (dict): Batch of images.
    Returns:
        texts (TextRecogDataSample): list of text predictions
    """
    model.eval()
    with torch.no_grad():
        inputs, data_sample = batch['inputs'], batch['data_samples']
        logits = model._forward(inputs, data_sample)
    return logits


def get_rec_model():
    config_path = '/home/shiyx/Projects/IC23/CoText/det_reg/configs/textrecog/unirec/unirec_dstext.py'
    ckpt_path = '/home/shiyx/Projects/IC23/CoText/det_reg/vit_base_pretrain_10ep_82.26/epoch_10.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path)

    return model


if __name__ == '__main__':
    config_path = 'configs/textrecog/unirec/unirec_dstext.py'
    ckpt_path = 'work_dirs/unirec_hiertext/vit_base_pretrain_10ep_82.26/epoch_10.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path).cuda()
    img = Image.open('88262.jpg')
    trans = T.ToTensor()
    img = trans(img).cuda()
    img_infos = [
        dict(
            img=img,
            polygons=[[0, 0, 16, 11], [0, 0, 16, 11], [0, 0, 16, 11],
                      [0, 0, 16, 11], [0, 0, 16, 11], [0, 0, 16, 11],
                      [0, 0, 16, 11], [0, 0, 16, 11]]),
        dict(
            img=img,
            polygons=[[0, 0, 16, 11], [0, 0, 16, 11], [0, 0, 16, 11],
                      [0, 0, 16, 11], [0, 0, 16, 11], [0, 0, 16, 11],
                      [0, 0, 16, 11], [0, 0, 16, 11]])
    ]

    batches = construct_batch(img_infos)
    results = []
    start = time.time()
    for batch in batches:
        logits = recog_forward_batch(model, batch)
        results.append(logits)
    print(time.time() - start)
    for result in results:
        print(result.shape)