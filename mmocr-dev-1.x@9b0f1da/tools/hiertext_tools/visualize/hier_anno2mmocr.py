import json

import mmengine
import numpy as np


def convert_anno(hiertext_anno_path: str):
    """Convert Prediction file to mmocr format

    Args:
        hiertext_anno_path (str): Path to the prediction file. The anno
            format is:
            {
                "annotations": [  // List of dictionaries, one for each image.
                    {
                    "image_id": "the filename of corresponding image.",
                    "paragraphs": [  // List of paragraphs.
                        {
                        "lines": [  // List of lines.
                            {
                            "text": "the text content of the entire line",
                            // Set to empty string for detection-only evaluation. # noqa: E501
                            "words": [  // List of words.
                                {
                                "vertices": [[x1, y1], [x2, y2],...,[xm, ym]],
                                "text": "the text content of this word",
                                // Set to empty string for detection-only evaluation. # noqa: E501
                                }, ...
                            ]
                            }, ...
                        ]
                        }, ...
                    ]
                    }, ...
                ]
                }
    """
    new_annos = {}
    with open(hiertext_anno_path, 'r') as f:
        predictions = json.loads(f.read().strip())['annotations']
    data_list = []
    for prediction in predictions:
        img_path = prediction['image_id']
        paragraphs = prediction['paragraphs']
        instances = []
        para_idx = 0
        line_idx = 0
        for paragraph in paragraphs:
            lines = paragraph['lines']
            for line in lines:
                words = line['words']
                for word in words:
                    text = word['text']
                    polygon = np.array(word['vertices']).reshape(-1).tolist()
                    instances.append(
                        dict(
                            text=text,
                            polygon=polygon,
                            para_id=para_idx,
                            line_id=line_idx,
                        ))
                line_idx += 1
            para_idx += 1
        data_list.append(dict(img_path=img_path, instances=instances))
    new_annos['data_list'] = data_list
    return new_annos


if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    new_annos = convert_anno(args.pred_path)
    mmengine.dump(new_annos, args.save_path)