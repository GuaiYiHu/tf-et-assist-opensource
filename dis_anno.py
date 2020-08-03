import logging
import sys
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

coco_pair = [
    (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def draw_human(img, anno, path):
    joints = []
    img = cv2.imread(
        path + '/pics/' + img['file_name'] + '.jpg')
    kp = np.array(anno['keypoints'])
    x = kp[0::3]
    y = kp[1::3]
    v = kp[2::3]
    for i, joint in enumerate(list(zip(x, y, v))):
        joints.append(joint)
    for pair_order, pair in enumerate(coco_pair):
        if joints[pair[0]][2] != 0 and joints[pair[1]][2] != 0 and pair is not (1, 3) and pair is not (
        2, 4) and pair_order not in range(0, 2):
            cv2.line(img, (int(joints[pair[0]][0] + 0.5), int(joints[pair[0]][1] + 0.5)),
                     (int(joints[pair[1]][0] + 0.5), int(joints[pair[1]][1] + 0.5)),
                     CocoColors[pair_order], 3)
    for i, joint in enumerate(list(zip(x, y, v))):
        if joint[2] != 0 and i not in range(1, 5):
            img = cv2.circle(img, (int(joint[0] + 0.5), int(joint[1] + 0.5)), 3, CocoColors[i], thickness=3, lineType=8,
                             shift=0)
    return img


def main_exp(all_video_list, video_parent_path, video_output_parent_path, exp_list, exp_type_list):
    action_list = all_video_list.keys()
    for i in exp_list:
        print(i)
        for action in action_list:
            for exp_type in exp_type_list:
                result_path = video_output_parent_path + '/' + i
                json_path = result_path + '/' + action + exp_type + '.json'
                video_save_path = result_path + '/' + action + exp_type + '.mp4'
                pic_path = result_path

                coco = COCO(json_path)
                video = video_parent_path + '/' + i + '/' + action + '.mp4'
                if video is not None:
                    cap = cv2.VideoCapture(video)
                else:
                    cap = cv2.VideoCapture(0)
                _, image = cap.read()
                if image is None:
                    logger.error("Can't read video")
                    sys.exit(-1)
                fps = cap.get(cv2.CAP_PROP_FPS)
                ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_saver = cv2.VideoWriter(video_save_path, fourcc, fps, (ori_w, ori_h))

                for img, anno in tqdm(list(zip(coco.dataset['images'], coco.dataset['annotations']))):
                    img = draw_human(img, anno, pic_path)
                    video_saver.write(img)


def main(all_video_list, video_output_parent_path, exp_type_list):
    action_list = all_video_list.keys()
    for action in action_list:
        video_list = all_video_list[action]
        for video in video_list:
            for exp_type in exp_type_list:
                dir_name = video.split('/')
                name = dir_name[-2]
                result_path = video_output_parent_path + '/' + name
                json_path = result_path + '/' + action + exp_type + '.json'
                video_save_path = result_path + '/' + action + exp_type + '.mp4'
                pic_path = result_path

                coco = COCO(json_path)

                if video is not None:
                    cap = cv2.VideoCapture(video)
                else:
                    cap = cv2.VideoCapture(0)
                _, image = cap.read()
                if image is None:
                    logger.error("Can't read video")
                    sys.exit(-1)
                fps = cap.get(cv2.CAP_PROP_FPS)
                ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_saver = cv2.VideoWriter(video_save_path, fourcc, fps, (ori_w, ori_h))

                for img, anno in tqdm(list(zip(coco.dataset['images'], coco.dataset['annotations']))):
                    img = draw_human(img, anno, pic_path)
                    video_saver.write(img)

if __name__ == '__main__':
    exp_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    video_parent_path = '/media/guaiyihu/diskD/dynamic/openpose_with_lstm_opensource/data'
    video_output_parent_path = '/media/guaiyihu/diskD/dynamic/openpose_with_lstm_opensource/out'

    action_list = ['平举']
    import get_video_list
    all_video_list = get_video_list.get_all_video_list(video_parent_path, action_list)
    main(all_video_list, video_output_parent_path, exp_list)
