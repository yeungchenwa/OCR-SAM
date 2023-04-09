import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile, Resize
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData

from mmocr.datasets.transforms import PackTextRecogInputs, Resize
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
    # checkpoint = torch.load(ckpt_path, map_location='cpu')
    # _load_checkpoint_to_model(model, checkpoint)
    return model


def construct_batch(data_infos):
    """Construct batch from data infos for Recognizer.

    Args:
        data_infos (list[dict]): List of data infos for Recognizer with the
            following keys:
                - img_path (str): Path to image file.
                - image_width (int): Width of image.
                - image_height (int): Height of image.
    """
    loading = LoadImageFromFile(color_type='color_ignore_orientation')
    resize = Resize(scale=(128, 32))
    packing = PackTextRecogInputs(
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    results = dict(inputs=[], data_samples=[])
    for data_info in data_infos:
        result = dict(img_path=data_info['img_path'])
        result = loading(result)
        result = resize(result)
        result = packing(result)
        results['inputs'].append(result['inputs'])
        results['data_samples'].append(result['data_samples'])
    return results


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
        batch = model.data_preprocessor(batch, training=False)
        inputs, data_sample = batch['inputs'], batch['data_samples']
        texts = model.predict(inputs, data_sample)
    return texts


if __name__ == '__main__':
    config_path = 'configs/textrecog/sar/sar_dstext.py'
    ckpt_path = 'work_dirs/unirec_hiertext/vit_base_pretrain_10ep_82.26/epoch_10.pth'
    register_all_modules(init_default_scope=True)

    model = build_model(config_path, ckpt_path)
    # batch size equals to 2
    data_infos = [
        dict(img_path='det_vis/0a3bc2f21ec1a7fc.jpg', ),
        dict(img_path='det_vis/0a3bc2f21ec1a7fc.jpg')
    ]
    batch = construct_batch(data_infos)
    texts = recog_forward_batch(model, batch)
    for text in texts:
        print(text.pred_text.item)
        print(text.pred_text.score)