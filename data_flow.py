import logging
import math
import os
import random
import struct

import cv2
import numpy as np
from pycocotools.coco import COCO
from tensorpack.dataflow.base import RNGDataFlow

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('data_flow')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class CocoMetadata:
    # __coco_parts = 57
    __coco_parts = 19
    __coco_vecs = list(zip(
        [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
        [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    ))

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(CocoMetadata.parse_float(four_nps[x * 4:x * 4 + 4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_url, img_meta, annotations, sigma):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        joint_list = []
        for ann in annotations:
            if ann.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        transform = list(zip(
            [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
            [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        ))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            new_joint.append((-1000, -1000))
            self.joint_list.append(new_joint)

        # logger.debug('joint size=%d' % len(self.joint_list))

    def get_heatmap(self, target_size):
        heatmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width), dtype=np.float32)

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, target_size):
        vectormap = np.zeros((CocoMetadata.__coco_parts * 2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width), dtype=np.int16)
        for joints in self.joint_list:
            for plane_idx, (j_idx1, j_idx2) in enumerate(CocoMetadata.__coco_vecs):
                j_idx1 -= 1
                j_idx2 -= 1

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
                    continue

                CocoMetadata.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p * 2 + 0] /= countmap[p][y][x]
            vectormap[y][x][p * 2 + 1] /= countmap[p][y][x]

        if target_size:
            vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        return vectormap.astype(np.float16)

    @staticmethod
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue

                countmap[plane_idx][y][x] += 1

                vectormap[plane_idx * 2 + 0][y][x] = vec_x
                vectormap[plane_idx * 2 + 1][y][x] = vec_y


class CocoPose(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        global mplset
        # if as_numpy and not mplset:
        #     import matplotlib as mpl
        #     mpl.use('Agg')
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(CocoPose.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = vectmap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, path, img_path=None, is_train=True, decode_img=True, only_idx=-1):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx

        if is_train:
            whole_path = os.path.join(path, 'person_keypoints_train2017.json')
        else:
            whole_path = os.path.join(path, 'person_keypoints_val2017.json')
        self.img_path = (img_path if img_path is not None else '') + ('train2017/' if is_train else 'val2017/')
        self.coco = COCO(whole_path)

        logger.info('%s dataset %d' % (path, self.size()))

    def size(self):
        return len(self.coco.imgs)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            # self.rng.shuffle(idxs)
            pass
        else:
            pass

        keys = list(self.coco.imgs.keys())
        for idx in idxs:
            img_meta = self.coco.imgs[keys[idx]]
            img_idx = img_meta['id']
            ann_idx = self.coco.getAnnIds(imgIds=img_idx)

            if 'http://' in self.img_path:
                img_url = self.img_path + img_meta['file_name']
            else:
                img_url = os.path.join(self.img_path, img_meta['file_name'])

            anns = self.coco.loadAnns(ann_idx)
            meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
            if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                continue

            yield [meta]
