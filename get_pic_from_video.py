import os

import cv2
from tqdm import tqdm


def save_pic(all_video_list, video_output_parent_path):
    action_list = all_video_list.keys()
    for action in action_list:
        video_list = all_video_list[action]
        for video in video_list:
            count = 0
            cap = cv2.VideoCapture(video)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 7
            r = w / h

            dir_name = video.split('/')
            name = dir_name[-2]

            save_img_path = video_output_parent_path + '/' + name + '/pics/'
            isExists = os.path.exists(save_img_path)

            if not isExists:
                os.makedirs(save_img_path)

            for _ in tqdm(range(frame_count)):
                if _ % 1 == 0:
                    _, frame = cap.read()
                    # frame_resized = cv2.resize(frame, (int(resize_h*r), resize_h))
                    # cv2.imshow('', frame)
                    # cv2.waitKey(1)
                    cv2.imwrite(save_img_path + action + '_%d.jpg' % count, frame)
                    count += 1


def main(all_video_list, video_output_parent_path):
    save_pic(all_video_list, video_output_parent_path)
