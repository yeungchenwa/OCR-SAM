import json
import os

from tqdm import tqdm


def Laion400M_convertor(file_path: str, out_dir: str):
    """Organize LAION400M dataset splits to a image-folder format

    Args:
        file_path (str): Path to a subset folder of LAION400M, and it should be 
            structured as follows:

            |__file_path
                |__00000
                    |__1.jpg
                    |__1.json
                    |__1.txt
                    |__2.jpg
                    |__2.json
                    |__2.txt
                    |__...
                |__00001
        out_dir (str): Path to the output, and it will be ordered in this way:
            |__out_dir
                |__imgs
                    |__1.jpg
                    |__2.jpg
                    |__...
                |__ori_annotation
                    |__annotation.jsonl
            
            The content in annotation.jsonl will be
                {"img_name": "1.jpg", "annotation": ...}
                {"img_name": "2.jpg", "annotation": ...}
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, 'imgs'))
        os.makedirs(os.path.join(out_dir, 'ori_annotation'))

    sub_folders = os.listdir(file_path)
    new_annos = []
    for i, sub_path in enumerate(sub_folders):
        if '.' in sub_path:
            continue
        print(f'Processing {sub_path} [{i +1}/{len(sub_folders)}]')
        files = os.listdir(os.path.join(file_path, sub_path))
        for file in tqdm(files):
            if not file.endswith('.jpg'):
                continue
            # move image to out_dir
            os.system(
                f'cp {os.path.join(file_path, sub_path, file)} {out_dir}/imgs')
            # load json
            ori_anno = json.load(
                open(
                    os.path.join(file_path, sub_path,
                                 file.replace('.jpg', '.json'))))
            new_annos.append(dict(img_name=file, annotation=ori_anno))
    with open(os.path.join(out_dir, 'ori_annotation/annotation.jsonl'),
              'w') as f:
        for new_anno in new_annos:
            f.write(json.dumps(new_anno, ensure_ascii=False))
            f.write('\n')


if __name__ == '__main__':
    Laion400M_convertor(
        '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00006',
        '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00006-reordered'
    )
