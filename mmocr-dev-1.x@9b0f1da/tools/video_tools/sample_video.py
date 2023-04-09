import random

import mmengine


def average_sample_frames(anno_path: str,
                          sample_freq: int,
                          ignore_empty_frame: bool = True):
    """Average Sample frames with sample_freq from each video

    Args:
        anno_path (str): Path to video annotation
        sample_freq (int): Sample frequency
        ignore_empty_frame (bool, optional): Whether to ignore empty frames.
    """
    anno = mmengine.load(anno_path)
    data_list = anno['data_list']
    new_data_list = []
    for i, data in enumerate(data_list):
        if ignore_empty_frame and len(data['instances']) == 0:
            continue
        if i % sample_freq == 0:
            new_data_list.append(data)
    print(f'Old num of frames: {len(anno["data_list"])}')
    print(f'New num of frames: {len(new_data_list)}')
    anno['data_list'] = new_data_list
    new_anno_path = anno_path.replace(
        '.json', f'_average_sample_freq_{sample_freq}.json')
    mmengine.dump(anno, new_anno_path)


def adjacent_sample_frames(anno_path: str, sample_nums: int):
    """Given a video, we divide the paragraphs according to the number of text
    instances each frame has. That is, frames with the same text instances are
    divided into the same paragraph. Next, we randomly draw frames in
    the same segment.

    Args:
        anno_path (str): Path to video annotation
        sample_nums (int): Sample number
    """
    anno = mmengine.load(anno_path)
    data_list = anno['data_list']
    video_id_split_data_list = {}
    for data in data_list:
        video_id = data['video_id']
        if video_id not in video_id_split_data_list:
            video_id_split_data_list[video_id] = []
        video_id_split_data_list[video_id].append(data)
    new_data_list = []
    zero_instances_frame = 0
    for video_id, data_list in video_id_split_data_list.items():
        # sort data_list by frame_id
        data_list = sorted(data_list, key=lambda x: x['frame_id'])
        # find all segments
        segments = [[]]
        current_segments_instances = -1
        for data in data_list:
            current_instances = len(data['instances'])
            if current_instances == 0:
                zero_instances_frame += 1
                continue
            if current_instances != current_segments_instances:
                segments.extend([[data]])
                current_segments_instances = current_instances
            else:
                segments[-1].append(data)
        # sample sample_nums from each segment
        for segment in segments[1:]:
            if len(segment) <= sample_nums:
                new_data_list.extend(segment)
            else:
                random.shuffle(segment)
                new_data_list.extend(segment[:sample_nums])
    print(f'zero instances frame: {zero_instances_frame}')
    print(f'Old num of frames: {len(anno["data_list"])}')
    print(f'New num of frames: {len(new_data_list)}')
    anno['data_list'] = new_data_list
    new_anno_path = anno_path.replace(
        '.json', f'_adjacent_sample_frames_{sample_nums}.json')
    mmengine.dump(anno, new_anno_path)


def sample_videos(anno_path: str, sample_nums: int):
    """Sample sample_nums of videos from annotaion

    Args:
        anno_path (str): Path to video annotation
        sample_nums (int): Sample number
    """
    anno = mmengine.load(anno_path)
    data_list = anno['data_list']
    # find all video ids
    video_ids = set()
    for data in data_list:
        video_ids.add(data['video_id'])
    # shuffle video ids and sample sample_nums of video ids
    video_ids = list(video_ids)
    random.shuffle(video_ids)
    video_ids = video_ids[:sample_nums]
    # find all data with video ids in video_ids
    new_data_list = []
    for data in data_list:
        if data['video_id'] in video_ids:
            new_data_list.append(data)
    print(f'Old num of frames: {len(anno["data_list"])}')
    print(f'New num of frames: {len(new_data_list)}')
    anno['data_list'] = new_data_list
    new_anno_path = anno_path.replace('.json',
                                      f'_sample_videos_{sample_nums}.json')
    mmengine.dump(anno, new_anno_path)


if __name__ == '__main__':
    anno_path = '/media/jiangqing/jqhard/DSText/annotation/det_annotations/val.json'
    sample_videos(anno_path, 10)