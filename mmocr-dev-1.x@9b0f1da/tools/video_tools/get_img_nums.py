import mmengine


def get_nums_det(anno_path: str):
    annos = mmengine.load(anno_path)
    data_list = annos['data_list']
    return len(data_list)


def get_nums_recog(anno_path: str):
    annos = mmengine.load(anno_path)
    data_list = annos['data_list']
    count = 0
    for data in data_list:
        count += len(data['instances'])
    return count


if __name__ == '__main__':
    print(
        get_nums_recog(
            '/media/jiangqing/jqhard/DSText/annotation/recog_annotations/train.json'
        ))
