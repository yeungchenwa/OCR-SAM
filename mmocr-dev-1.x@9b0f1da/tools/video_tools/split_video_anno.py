import os.path as osp
import random

import mmengine


def split_anno(anno_path: str, output_dir: str):
    """Split video annotation file into multiple files based on video name.

    Args:
        anno_path (str): Path to video annotation file.
        output_dir (str): Output directory.
    """
    video_annos = mmengine.load(anno_path)
    meta_info = video_annos['metainfo']
    data_list = video_annos['data_list']
    splits = {}
    for data in data_list:
        video_name = data['img_path'].split('/')[0]
        if video_name not in splits:
            splits[video_name] = []
        splits[video_name].append(data)
    for video_name, data in splits.items():
        video_annos = dict(metainfo=meta_info, data_list=data)
        output_path = osp.join(output_dir, f'{video_name}.json')
        mmengine.dump(video_annos, output_path)


def split_val_recog(
    anno_path: str,
    output_dir: str = None,
    val_nums: int = 20000,
):
    """Split part of the validation set from the original annotation file to
    speed up the eval process.

    Args:
        anno_path (str): Path to video annotation file.
        val_nums (int): Number of videos in the validation set.
        output_dir (str): Output directory.
    """
    val_annos = mmengine.load(anno_path)
    meta_info = val_annos['metainfo']
    data_list = val_annos['data_list']
    splits = []
    # shuffle the data_list
    random.shuffle(data_list)
    total_instances = 0
    for data in data_list:
        total_instances += len(data['instances'])
        splits.append(data)
        if total_instances >= val_nums:
            break
    new_data_list = splits
    val_annos = dict(metainfo=meta_info, data_list=new_data_list)
    out_name = f'val_{val_nums}_instances.json'
    mmengine.dump(val_annos, osp.join(output_dir, out_name))


if __name__ == '__main__':
    split_val_recog(
        '/media/jiangqing/jqhard/DSText/annotation/recog_annotations/val.json',
        '/media/jiangqing/jqhard/DSText/annotation/recog_annotations/')
