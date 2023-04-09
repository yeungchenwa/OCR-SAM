import os

import mmengine
from tools.hiertext_tools.end2end import (construct_pseudo_det_annos,
                                          construct_pseudo_recog_annos,
                                          crop_text_instances,
                                          merge_det_recog_results,
                                          visualize_e2e)
from tools.video_tools.visualize_det import visualize_det

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def modify_det_config(img_dirs: str, work_dir: str):
    configs = mmengine.Config.fromfile(
        'configs/textdet/_base_/datasets/laion400m.py')
    configs['laion400m_textdet_test']['data_prefix']['img_path'] = img_dirs
    configs['laion400m_textdet_test']['ann_file'] = os.path.join(
        work_dir, 'laion400m_pseudo_det.json')
    mmengine.Config.dump(configs,
                         'configs/textdet/_base_/datasets/laion400m.py')


def modify_recog_config(work_dir: str):
    configs = mmengine.Config.fromfile(
        'configs/textrecog/_base_/datasets/dstext.py')
    configs['dstext_textrecog_test']['ann_file'] = os.path.join(
        work_dir, 'laion400m_pseudo_recog.json')
    mmengine.Config.dump(configs,
                         'configs/textrecog/_base_/datasets/dstext.py')


if __name__ == '__main__':
    work_dir = 'work_dirs/LAION400M/part-00000'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # videos_root = 'data/DSText/det_images/test'
    img_dirs = '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00000-reordered/imgs'
    current_work_dir = os.path.join(work_dir)
    modify_det_config(img_dirs, current_work_dir)
    print(f'Constructing pseudo annotations...')
    construct_pseudo_det_annos(
        img_dirs, os.path.join(current_work_dir, 'laion400m_pseudo_det.json'))
    print(f'Predicting text instances for...')
    os.system(
        f'python tools/test.py configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k_LAION400M.py checkpoints/db_swin_mix_pretrain.pth --save-preds --work-dir {current_work_dir}/det_out'
    )
    print(f'Croping text instances for...')
    crop_text_instances(
        os.path.join(current_work_dir, 'det_out',
                     'db_swin_mix_pretrain.pth_predictions.pkl'),
        recog_out_dir=os.path.join(current_work_dir, 'recog_crop'),
        use_affine_crop=False)

    print(f'Constructing recog annotations for ...')
    construct_pseudo_recog_annos(
        os.path.join(current_work_dir, 'recog_crop'),
        os.path.join(current_work_dir, 'laion4m_pseudo_recog.json'))
    print(f'Predicting text recognition for ...')
    modify_recog_config(current_work_dir)
    os.system(
        f'python tools/test.py configs/textrecog/unirec/unirec_dstext.py checkpoints/unirec.pth --work-dir {current_work_dir}/recog_out'
    )
    print('merging results ...')
    merge_det_recog_results(
        os.path.join(current_work_dir, 'det_out',
                     'db_swin_mix_pretrain.pth_predictions.pkl'),
        os.path.join(current_work_dir, 'recog_out',
                     'unirec.pth_predictions.pkl'),
        os.path.join(current_work_dir, 'e2e.pkl'),
        min_area=14 * 14)
