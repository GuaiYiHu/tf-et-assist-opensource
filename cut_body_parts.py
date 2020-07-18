import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


class cut_body_part:
    def __init__(self, anno_file, coco_images, img_path=None, save_path=None):
        self.coco_anno_train2017 = '/Volumes/data-1/yzy/dynamic/dataset/annotations/' \
                                   'person_keypoints_train2017.json'
        self.coco_anno_val2017 = '/Volumes/data-1/yzy/dynamic/dataset/annotations/' \
                                 'person_keypoints_val2017.json'
        # self.coco_images = '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/caculate_distance/'
        self.coco_images = coco_images
        self.img_path = img_path
        self.save_path = save_path
        # self.hand_images = '/Volumes/data-1/yzy/dynamic/dataset/images/hand/'
        # coco_anno_train2017 = '/Volumes/data-1/yzy/dataset/Realtime_Multi-Person_Pose_Estimation-master/' \
        #                       'training/dataset/COCO/annotations/person_keypoints_train2017.json'
        # coco_anno_val2017 = '/Volumes/data-1/yzy/dataset/Realtime_Multi-Person_Pose_Estimation-master/' \
        #                       'training/dataset/COCO/annotations/person_keypoints_val2017.json'
        # coco_images = '/Volumes/data-1/yzy/dataset/Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/'
        # self.coco = COCO(self.coco_anno_train2017)
        self.coco = COCO(anno_file)

    def size(self):
        return len(self.coco.imgs)

    def crop_part(self):
        for img_meta, anno in tqdm(list(zip(self.coco.dataset['images'], self.coco.dataset['annotations']))):
            result = []
            assert img_meta['id'] == anno['image_id']
            img_name = img_meta['file_name']
            # keypoints = anno['keypoints']

            img = cv2.imread(self.coco_images + img_name + '.jpg')
            assert img is not None
            kp = np.array(anno['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]
            l = []
            r = []
            hands_distance = np.sqrt(np.square(x[9] - x[10]) + np.square(y[9] - y[10]))
            if v[7] == 1 and v[9] == 1:
                distance = np.sqrt(np.square(x[7] - x[9]) + np.square(y[7] - y[9]))
                distance = min([distance, hands_distance])
                dis = int(distance * 2 + 0.5 if distance <= 40 else distance / 2 + 0.5)
                left = x[9] - dis
                top = y[9] - dis
                right = x[9] + dis
                down = y[9] + dis
                try:
                    croped = img[top:down + 1, left:right + 1, :]
                    tmp = np.zeros([max(croped.shape[0], croped.shape[1]), max(croped.shape[0], croped.shape[1]), 3],
                                   dtype=np.uint8)
                    tmp[0:croped.shape[0], 0:croped.shape[1], :] = croped
                    croped = cv2.resize(tmp, (368, 368))
                    l = copy.deepcopy(croped)
                    result.append({'idx': 9, 'hand': croped, 'position': [left, top, right, down]})
                except BaseException:
                    pass
            if v[8] == 1 and v[10] == 1:
                distance = np.sqrt(np.square(x[8] - x[10]) + np.square(y[8] - y[10]))
                dis = int(distance * 3 + 0.5 if distance <= 40 else distance / 2 + 0.5)
                left = x[10] - dis
                top = y[10] - dis
                right = x[10] + dis
                down = y[10] + dis
                try:
                    croped = img[top:down + 1, left:right + 1, :]
                    tmp = np.zeros([max(croped.shape[0], croped.shape[1]), max(croped.shape[0], croped.shape[1]), 3],
                                   dtype=np.uint8)
                    tmp[0:croped.shape[0], 0:croped.shape[1], :] = croped
                    croped = cv2.resize(tmp, (368, 368))
                    r = copy.deepcopy(croped)
                    result.append({'idx': 10, 'hand': croped, 'position': [left, top, right, down]})
                except BaseException:
                    pass
            try:
                both_hand = np.concatenate((r, l), axis=1)
                # cv2.imshow('', both_hand)
                # video_saver.write(both_hand)
                # cv2.waitKey(1)
            except:
                pass
            yield img, result, img_meta, anno

    def get_data(self):
        idxs = np.arange(self.size())
        # if self.is_train:
        #     self.rng.shuffle(idxs)
        #
        # else:
        #     pass
        keys = list(self.coco.imgs.keys())
        count = 0
        for idx in tqdm(idxs):
            img_meta = self.coco.imgs[keys[idx]]
            img_idx = img_meta['id']
            ann_idx = self.coco.getAnnIds(imgIds=img_idx)
            img_url = os.path.join(self.img_path, img_meta['file_name'])
            raw_img = cv2.imread(img_url)
            anns = self.coco.loadAnns(ann_idx)

            for ann in anns:
                if ann['iscrowd']:
                    continue
                mask_img = copy.deepcopy(raw_img)
                for other_ann in anns:
                    if other_ann == ann:
                        continue
                    tmp_mask = self.coco.annToMask(other_ann)
                    miss_mask = np.zeros((img_meta['height'], img_meta['width'])).astype(np.bool)
                    miss_mask = np.invert(np.logical_or(miss_mask, tmp_mask)[:, :, np.newaxis])
                    mask_img = mask_img * miss_mask
                # cv2.imshow('', mask_img)
                # cv2.waitKey(1)
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                hands_distance = np.sqrt(np.square(x[9] - x[10]) + np.square(y[9] - y[10]))
                if v[7] in [1, 2] and v[9] == 2:
                    distance = (np.sqrt(np.square(x[7] - x[9]) + np.square(y[7] - y[9])))
                    # dis = int(distance * 3 + 0.5 if distance <= 40 else distance + 0.5)
                    dis = int(distance + 0.5)
                    left = x[9] - dis
                    top = y[9] - dis
                    right = x[9] + dis
                    down = y[9] + dis
                    if dis < hands_distance and dis >= 20:
                        try:
                            croped = cv2.resize(mask_img[top:down + 1, left:right + 1, :], (368, 368))
                            # l = copy.deepcopy(croped)
                            # result.append({'idx': 9, 'hand': croped, 'position': [left, top, right, down]})
                            file_name = self.save_path + str(count).rjust(8, '0') + '.jpg'
                            # cv2.imshow('l', croped)
                            # cv2.waitKey(1)
                            cv2.imwrite(file_name, croped)
                            count += 1
                            print(file_name)
                        except BaseException:
                            pass
                if v[8] in [1, 2] and v[10] == 2:
                    distance = (np.sqrt(np.square(x[8] - x[10]) + np.square(y[8] - y[10])))
                    # dis = int(distance * 3 + 0.5 if distance <= 40 else distance + 0.5)
                    dis = int(distance + 0.5)
                    left = x[10] - dis
                    top = y[10] - dis
                    right = x[10] + dis
                    down = y[10] + dis
                    if dis < hands_distance and dis >= 20:
                        try:
                            croped = cv2.resize(mask_img[top:down + 1, left:right + 1, :], (368, 368))
                            # r = copy.deepcopy(croped)
                            # result.append({'idx': 10, 'hand': croped, 'position': [left, top, right, down]})
                            file_name = self.save_path + str(count).rjust(8, '0') + '.jpg'
                            # cv2.imshow('r', croped)
                            # cv2.waitKey(1)
                            cv2.imwrite(file_name, croped)
                            count += 1
                            print(file_name)
                        except BaseException:
                            pass
                # meta = CocoMetadata(idx, img_url, img_meta, anns, np.invert(miss_mask[:, :, np.newaxis]), sigma=8.0)
                #
                # total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
                # if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                #     continue

                yield mask_img


if __name__ == '__main__':
    cut_body_part = cut_body_part('/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
                                  'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/'
                                  'person_keypoints_train2017.json',
                                  '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
                                  'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/train2017/',
                                  save_path='/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
                                            'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/hand/')
    for img in cut_body_part.get_data():
        pass
