import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm


def vis_submission(imgs_dir, xml_path, vis_dir):
    """Visualize submission xml file

    Args:
        imgs_dir (str): path to images directory
        xml_path (str): path to xml file
        vis_dir (str): path to visualization directory
    
        The structure of xml file is like:
            <?xml version="1.0" ?>
            <Frames>
                <frame ID="1">
                    <object ID="16" Transcription="RUYLAND">
                    <Point x="1026" y="405"/>
                    <Point x="1026" y="390"/>
                    <Point x="1065" y="390"/>
                    <Point x="1065" y="405"/>
                    </object>
                    <object ID="8" Transcription="E">
                    <Point x="107" y="899"/>
                    <Point x="179" y="895"/>
                    <Point x="180" y="912"/>
                    <Point x="109" y="916"/>
                    </object>
                </frame>
                <frame ID="2">
                    <object ID="23" Transcription="E">
                    <Point x="1417" y="692"/>
                    <Point x="1417" y="688"/>
                    <Point x="1430" y="688"/>
                    <Point x="1430" y="692"/>
                    </object>
    """
    # load xml file
    COLORS = np.random.uniform(0, 255, size=(2000, 3))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # create vis_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # get all the Frames
    frames = root.findall('frame')
    # visualize each frame
    for frame in tqdm(frames):
        # get frame id
        frame_id = frame.attrib['ID']
        # get all the objects
        objects = frame.findall('object')
        # read image
        img_path = os.path.join(imgs_dir, frame_id + '.jpg')
        img = cv2.imread(img_path)
        # draw each object
        for obj in objects:
            transcription = obj.attrib['Transcription']
            points = obj.findall('Point')
            id = int(obj.attrib['ID'])
            pts = []
            for point in points:
                x = int(point.attrib['x'])
                y = int(point.attrib['y'])
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, COLORS[id], 4)
            cv2.putText(img, transcription, (pts[0][0][0], pts[0][0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # save image
        vis_img_path = os.path.join(vis_dir, frame_id + '.jpg')
        cv2.imwrite(vis_img_path, img)


def get_recog_submission(det_submission_root):
    """Creat Submission file for e2e recognition

    Args:
        det_submission_root (str): path to det_submission directory
    """
    xml_files = os.listdir(det_submission_root)
    for xml_file in xml_files:
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(det_submission_root, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frames = root.findall('frame')
        # creat a target dict
        object_id_dict = {}
        for frame in frames:
            objects = frame.findall('object')
            for obj in objects:
                transcription = obj.attrib['Transcription']
                object_id = int(obj.attrib['ID'])
                if object_id not in object_id_dict:
                    object_id_dict[object_id] = []
                object_id_dict[object_id].append(transcription)
        # for each id, get the most frequent transcription
        for object_id in object_id_dict:
            transcriptions = object_id_dict[object_id]
            unique, counts = np.unique(transcriptions, return_counts=True)
            transcription = unique[np.argmax(counts)]
            object_id_dict[object_id] = transcription
        # save to txt
        txt_path = xml_path.replace('.xml', '.txt')
        with open(txt_path, 'w') as f:
            # sort by object_id
            object_id_dict = dict(
                sorted(object_id_dict.items(), key=lambda item: item[0]))
            for object_id in object_id_dict:
                transcription = object_id_dict[object_id]
                target_str = '"' + str(
                    object_id) + '"' + ',' + '"' + transcription + '"' + '\n'
                f.write(target_str)


def check_recog_result(txt_dir, xml_dir):
    error_num = 0
    xml_list = os.listdir(xml_dir)
    for xml_file in xml_list:
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        txt_file = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(txt_dir, txt_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # get all the Frames
        frames = root.findall('frame')
        with open(txt_path, 'r') as fr:
            lines = fr.readlines()
        for frame in tqdm(frames):
            # get frame id
            frame_id = frame.attrib['ID']
            # get all the objects
            objects = frame.findall('object')
            # draw each object
            for obj in objects:
                transcription = obj.attrib['Transcription']
                id = int(obj.attrib['ID'])
                txt_transcription = lines[int(id)].split(',')[-1][1:-2]
                if transcription != txt_transcription:
                    print(xml_file + ' ' + str(frame_id) + ' ' + str(id))
                    error_num += 1
    print(error_num)


if __name__ == '__main__':
    # imgs_dir, xml_path, vis_dir
    # video_dir = 'data/DSText/det_images/test'
    # submission_dir = 'dstext_submission/det_submission'
    # visualization_dir = 'vis'
    # vid_list = os.listdir(video_dir)
    # for video in vid_list:
    #     imgs_dir = os.path.join(video_dir, video)
    #     xml_dir = os.path.join(submission_dir, 'res_'+video+'.xml')
    #     vis_dir = os.path.join(visualization_dir, 'res_'+video)
    #     vis_submission(imgs_dir, xml_dir, vis_dir)

    txt_dir = 'dstext_submission/e2e_det_submission'
    xml_dir = 'dstext_submission/e2e_det_submission'
    check_recog_result(txt_dir, xml_dir)

    # vis_submission('data/DSText/det_images/test/Video_105_0_5',
    #                'dstext_submission/det_submission/res_Video_105_0_5.xml',
    #                'vis/res_Video_105_0_5')
