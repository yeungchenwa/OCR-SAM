import os
import json
from tqdm import tqdm

def find_candidates(e2e_path:str, ori_anno_path:str, out_anno_path:str):
    """Find instances with captions in the e2e results

    Args:
        e2e_path (str): Path to e2e results
        ori_anno_path (str): Path to original annotation
        out_anno_path (str): Path to save filtered annotation
    """
    with open(e2e_path, 'r') as f:
        e2e_annos = [json.loads(line) for line in f.readlines()]
        # sort by img_path
        e2e_annos = sorted(e2e_annos, key=lambda x: x['img_path'].split('/')[-1].split('.')[0])
    with open(ori_anno_path, 'r') as f:
        ori_annos = [json.loads(line) for line in f.readlines()]
        # sort by img_name
        ori_annos = sorted(ori_annos, key=lambda x: x['img_name'].split('.')[0])
    ori_annos = {anno['img_name']: anno for anno in ori_annos}
    new_annos = []
    for e2e_anno in tqdm(e2e_annos):
        img_name = e2e_anno['img_path'].split('/')[-1]
        ori_anno = ori_annos[img_name]
        caption = ori_anno['annotation']['caption']
        if caption == None:
            continue
        words = [word.lower() for word in caption.split(' ')]
        pred_text = e2e_anno['text']
        if pred_text == '':
            continue
        pred_words = [word.lower() for word in pred_text]
        # creat a position map
        pos_map = {}
        for i, pred_word in enumerate(pred_words):
            if pred_word in words:
                pos_map[i] = words.index(pred_word)
        if len(pos_map) == 0:
            continue
        else:
            new_anno = {
                'img_name': img_name,
                'img_path': e2e_anno['img_path'],
                'polygons': e2e_anno['polygons'],
                'det_scores': e2e_anno['scores'],
                'pred_text': pred_text,
                'rec_scores': e2e_anno['text_confidence'],
                'caption': caption,
                'pos_map': pos_map
            }
            new_annos.append(new_anno)
    with open(out_anno_path, 'w') as f:
        for anno in new_annos:
            f.write(json.dumps(anno) + '\n')
    print(f'Find {len(new_annos)} instances with captions in the e2e results.')
           
def analysis_candidates(candidate_path:str):
    """Analysis the length of the candidates

    Args:
        candidate_path (str): Path to the candidates
    """
    with open(candidate_path, 'r') as f:
        annos = [json.loads(line) for line in f.readlines()]
    length_map = {}
    for anno in annos:
        pos_map = anno['pos_map']
        length = len(pos_map)
        if length not in length_map:
            length_map[length] = 1
        else:
            length_map[length] += 1
    # sort by length_map by key
    length_map = dict(sorted(length_map.items(), key=lambda x: x[0]))
    for length, count in length_map.items():
        print(f'Length: {length}, Count: {count}, ratio: {count/len(annos)}')
    total_candidates = sum(length_map.values())
    # plot the ratio distribution
    import matplotlib.pyplot as plt
    plt.bar(length_map.keys(), [count/len(annos) for count in length_map.values()])
    # restrict the x-axis
    plt.xlim(0, 20)
    # the x axis should be integer
    plt.xticks(range(0, 21, 1))
    # x and y labels
    plt.xlabel('Length')
    plt.ylabel('Ratio')
    # set total_candidates as legend
    plt.legend([f'Total Candidates: {total_candidates}'])
    plt.show()
    # save the length distribution
    plt.savefig('length_distribution.png')

if __name__ == '__main__':
   # ori_anno_path = '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00000-reordered/ori_annotation/annotation.jsonl'
    #e2e_path = '/media/jiangqing/jqssd/projects/research/TextCLIP/mmocr-dev-1.x@9b0f1da/work_dirs/LAION400M/part-00000/e2e.jsonl'
    out_anno_path = '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00000-reordered/ori_annotation/candidates.jsonl'
    #find_candidates(e2e_path, ori_anno_path, out_anno_path)
    analysis_candidates(out_anno_path)