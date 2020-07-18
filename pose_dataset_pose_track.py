import copy
import json
import logging
import multiprocessing
import struct
import sys
import threading

from matplotlib.font_manager import FontProperties
from scipy import signal

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import requests
import numpy as np
import time

import tensorflow as tf
from skimage.draw import polygon
from tensorpack.dataflow import MultiThreadMapData
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.parallel import PrefetchData, PrefetchDataZMQ
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from common import cross_keypoints_list
from pycocotools.coco import COCO
from matplotlib import pyplot as plt

plt.interactive(False)
import math
import random

import cv2
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid

from common import CocoPart, parkinson_cut_list

_network_w = 368
_network_h = 368
_scale = 8

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mplset = False
font = FontProperties(fname=r"simsun.ttc", size=20)


def read_image_path(img_path):
    img_str = open(img_path, 'rb').read()
    if not img_str:
        logger.warning('image not read, path=%s' % img_path)
        raise Exception()
    nparr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    return img, height, width


def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h


def set_network_scale(scale):
    global _scale
    _scale = scale


def resize_img(meta):
    global _network_w, _network_h
    img = meta.img
    r = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (_network_w, int(_network_w * r)), interpolation=cv2.INTER_AREA)
    meta.img = img
    return meta


def pose_scale_fixed(meta):
    global _network_w, _network_h
    scale = _network_h / meta.height if _network_h / meta.height >= _network_w / meta.width else _network_w / meta.width
    neww = int(meta.width * scale)
    newh = int(meta.height * scale)
    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0] * scalew + 0.5) > neww or int(
            #                         point[1] * scaleh + 0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scale + 0.5), int(point[1] * scale + 0.5)))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta


def pose_random_scale(metas):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    for meta in metas:
        neww = int(meta.width * scalew)
        newh = int(meta.height * scaleh)
        dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

        # adjust meta data
        adjust_joint_list = []
        for joint in meta.joint_list:
            adjust_joint = []
            for point in joint:
                if point[0] < -100 or point[1] < -100:
                    adjust_joint.append((-1000, -1000))
                    continue
                # if point[0] <= 0 or point[1] <= 0 or int(point[0] * scalew + 0.5) > neww or int(
                #                         point[1] * scaleh + 0.5) > newh:
                #     adjust_joint.append((-1, -1))
                #     continue
                adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
            adjust_joint_list.append(adjust_joint)

        meta.joint_list = adjust_joint_list
        meta.width, meta.height = neww, newh
        meta.img = dst
    return metas


def smoother(path):
    data = json.load(open(path, 'r'))
    draw_human_data = copy.deepcopy(data)
    fps = 30
    keypoint_index = 31
    points = []
    for frame in data['annotations']:
        points.append(frame['keypoints'])
    points = np.array(points)
    # points[:, 0::3] = np.true_divide(points[:, 0::3], data['images'][0]['width'])
    # points[:, 1::3] = np.true_divide(points[:, 1::3], data['images'][0]['height'])
    b, a = signal.butter(2, [0.1, 0.4], 'bandpass')
    sf = signal.filtfilt(b, a, points, axis=0)
    for i, keypoints in enumerate(sf):
        data['annotations'][i]['keypoints'] = list(keypoints)
    ref = {"images": data['images'], "annotations": data['annotations'], "LUE": 1, "RUE": 4}
    save_path = path.split('.')[0] + '_smooth.json'
    with open(save_path, "w") as f:
        json.dump(ref, f)
        print('writed to ' + save_path)

    return save_path


def pose_resize_shortestedge_fixed(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = max(ratio_w, ratio_h)
    return pose_resize_shortestedge(meta, int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5)))


def pose_resize_shortestedge_random(metas):
    ratio_w = _network_w / metas[0].width
    ratio_h = _network_h / metas[0].height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(metas[0].width * ratio + 0.5, metas[0].height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.95, 1.6))
    return pose_resize_shortestedge(metas, target_size)


def pose_resize_shortestedge(metas, target_size):
    global _network_w, _network_h

    # adjust image
    scale = target_size / min(metas[0].height, metas[0].width)
    color = random.randint(0, 255)
    if metas[0].height < metas[0].width:
        newh, neww = target_size, int(scale * metas[0].width + 0.5)
    else:
        newh, neww = int(scale * metas[0].height + 0.5), target_size
    for meta in metas:
        img = meta.img
        dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

        pw = ph = 0
        if neww < _network_w or newh < _network_h:
            pw = max(0, (_network_w - neww) // 2)
            ph = max(0, (_network_h - newh) // 2)
            mw = (_network_w - neww) % 2
            mh = (_network_h - newh) % 2

            dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(color, 0, 0))

        # adjust meta data
        adjust_joint_list = []
        for joint in meta.joint_list:
            adjust_joint = []
            for point in joint:
                if point[0] < -100 or point[1] < -100:
                    adjust_joint.append((-1000, -1000))
                    continue
                # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
                #     adjust_joint.append((-1, -1))
                #     continue
                adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
            adjust_joint_list.append(adjust_joint)

        meta.joint_list = adjust_joint_list
        meta.width, meta.height = neww + pw * 2, newh + ph * 2
        meta.img = dst
    return metas


def pose_crop_center(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    x = (meta.width - target_size[0]) // 2 if meta.width > target_size[0] else 0
    y = (meta.height - target_size[1]) // 2 if meta.height > target_size[1] else 0

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop_random(metas):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)

    for _ in range(50):
        x = random.randrange(0, metas[0].width - target_size[0]) if metas[0].width > target_size[0] else 0
        y = random.randrange(0, metas[0].height - target_size[1]) if metas[0].height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets
        for joint in metas[0].joint_list:
            if x <= joint[CocoPart.Nose.value][0] < x + target_size[0] and y <= joint[CocoPart.Nose.value][1] < y + \
                    target_size[1]:
                break

    return pose_crop(metas, x, y, target_size[0], target_size[1])


def pose_crop_fixed(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    x = 0
    y = 0
    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop(metas, x, y, w, h):
    # adjust image
    target_size = (w, h)
    for meta in metas:
        img = meta.img
        resized = np.zeros([target_size[1], target_size[0], img.shape[2]])
        croped = img[y:y + target_size[1], x:x + target_size[0], :]
        resized[:croped.shape[0], :croped.shape[1], :] = croped
        # adjust meta data
        adjust_joint_list = []
        for joint in meta.joint_list:
            adjust_joint = []
            for point in joint:
                if point[0] < -100 or point[1] < -100:
                    adjust_joint.append((-1000, -1000))
                    continue
                # if point[0] <= 0 or point[1] <= 0:
                #     adjust_joint.append((-1000, -1000))
                #     continue
                new_x, new_y = point[0] - x, point[1] - y
                # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
                #     adjust_joint.append((-1, -1))
                #     continue
                adjust_joint.append((new_x, new_y))
            adjust_joint_list.append(adjust_joint)

        meta.joint_list = adjust_joint_list
        meta.width, meta.height = target_size
        meta.img = resized
    return metas


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    flip_list = [CocoPart.Nose, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder,
                 CocoPart.RElbow, CocoPart.RWrist,
                 CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle,
                 CocoPart.LEye, CocoPart.REye, CocoPart.LEar, CocoPart.REar, CocoPart.Background]
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for cocopart in flip_list:
            point = joint[cocopart.value]
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((meta.width - point[0], point[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_rotation(metas):
    deg = random.uniform(-15.0, 15.0)
    for meta in metas:
        img = meta.img

        center = (img.shape[1] * 0.5, img.shape[0] * 0.5)  # x, y
        rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
        neww = min(neww, ret.shape[1])
        newh = min(newh, ret.shape[0])
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        # print(ret.shape, deg, newx, newy, neww, newh)
        img = ret[newy:newy + newh, newx:newx + neww]

        # adjust meta data
        adjust_joint_list = []
        for joint in meta.joint_list:
            adjust_joint = []
            for point in joint:
                if point[0] < -100 or point[1] < -100:
                    adjust_joint.append((-1000, -1000))
                    continue
                # if point[0] <= 0 or point[1] <= 0:
                #     adjust_joint.append((-1, -1))
                #     continue
                x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
                adjust_joint.append((x, y))
            adjust_joint_list.append(adjust_joint)

        meta.joint_list = adjust_joint_list
        meta.width, meta.height = neww, newh
        meta.img = img

    return metas


def pose_gen_mask(metas):
    # plt.imshow(meta.img)
    # plt.show()
    for meta in metas:
        if 'ignore_regions_x' in meta.meta:
            for region_x, region_y in zip(meta.meta['ignore_regions_x'], meta.meta['ignore_regions_y']):
                rr, cc = polygon(region_y, region_x, meta.img.shape)
                meta.img[rr, cc, :] = 0
        # plt.imshow(meta.img)
        # plt.show()

    return metas


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)


def pose_to_img(meta_ls):
    global _network_w, _network_h, _scale
    img = None
    heatmap = None
    vectormap = None
    for meta_l in meta_ls:
        if img is None:
            img = meta_l.img.astype(np.float16)[np.newaxis, :, :, :]
            heatmap = meta_l.get_heatmap(target_size=(_network_w // _scale, _network_h // _scale))[np.newaxis, :, :, :]
            vectormap = meta_l.get_vectormap(target_size=(_network_w // _scale, _network_h // _scale))[np.newaxis, :, :,
                        :]
        else:
            img = np.concatenate((img, meta_l.img.astype(np.float16)[np.newaxis, :, :, :]), axis=0)
            heatmap = np.concatenate((heatmap,
                                      meta_l.get_heatmap(target_size=(_network_w // _scale, _network_h // _scale))[
                                      np.newaxis, :, :, :]), axis=0)
            vectormap = np.concatenate((vectormap,
                                        meta_l.get_vectormap(target_size=(_network_w // _scale, _network_h // _scale))[
                                        np.newaxis, :, :, :]), axis=0)
    assert img is not None
    assert heatmap is not None
    assert vectormap is not None

    return img, heatmap, vectormap


class tfrecods_writer():
    def __init__(self, train_filename):
        self.writer = tf.python_io.TFRecordWriter(train_filename)

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def new_example(self, img_num, imgs, heatmaps, pafs):
        feature = {
            'img_num': self._bytes_feature(tf.compat.as_bytes(img_num.tostring())),
            'imgs': self._bytes_feature(tf.compat.as_bytes(imgs.tostring())),
            'heatmaps': self._bytes_feature(tf.compat.as_bytes(heatmaps.tostring())),
            'pafs': self._bytes_feature(tf.compat.as_bytes(pafs.tostring()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

    def finishi_write(self):
        self.writer.close()


class CocoMetadata:
    # __coco_parts = 16
    __coco_parts = 15
    # __coco_vecs = list(zip(
    #     [2, 9,  10, 2,  12, 13, 2, 3, 4, 2, 6, 7, 2, 1, 2],
    #     [9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8, 1, 15, 15]
    # ))
    __coco_vecs = list(zip(
        [2, 9, 10, 2, 12, 13, 2, 3, 4, 2, 6, 7, 2],
        [9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8, 1]
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
        self.annotations = annotations
        self.meta = img_meta
        self.sigma = sigma
        # self.annos = annotations

        # self.height = int(img_meta['height'])
        # self.width = int(img_meta['width'])

        self.height = int(0)
        self.width = int(0)

        joint_list = []
        for ann in annotations:
            # if ann.get('num_keypoints', 0) == 0:
            #     continue

            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            joint_list.append([(x, y) if x > 0 or y > 0 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        # transform = list(zip(
        #     [1, 2, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3],
        #     [1, 2, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3]
        # ))
        transform = list(zip(
            [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16],
            [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16]
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
        assert self.height != 0
        assert self.width != 0
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
        vectormap = np.zeros((CocoMetadata.__coco_parts * 2 - 4, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((CocoMetadata.__coco_parts - 2, self.height, self.width), dtype=np.int16)
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


class PoseTrackPose(RNGDataFlow):
    @staticmethod
    def is_json(name):
        return name.split('.')[-1] == 'json'

    def __init__(self, path, img_path=None, is_train=True, decode_img=True, only_idx=-1, image_num=5):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.image_num = image_num
        if is_train:
            self.whole_path = path
        else:
            self.whole_path = path
        self.img_path = img_path
        self.anno_list_raw = filter(self.is_json, os.listdir(self.whole_path))
        self.anno_list = [iter_var for iter_var in self.anno_list_raw]
        # self.coco = COCO(whole_path)

        logger.info('%s dataset %d' % (path, self.size()))

    def size(self):
        return len(self.anno_list)

    def get_data(self):
        anno_file_idxs = np.arange(self.size())
        self.rng.shuffle(anno_file_idxs)
        for anno_file_idx in anno_file_idxs:
            metas = []
            coco = COCO(self.whole_path + self.anno_list[anno_file_idx], show_log=False)
            keys = list(coco.imgs.keys())
            while 1:
                idx_low = random.randint(0, len(coco.imgs) - self.image_num)
                img_meta_low = coco.imgs[keys[idx_low]]
                img_meta_high = coco.imgs[keys[idx_low + self.image_num - 1]]
                if img_meta_low['is_labeled'] and img_meta_high['is_labeled']:
                    break
            idxs = [i + idx_low for i in range(self.image_num)]
            # idxs = np.arange(self.size())
            # if self.is_train:
            #     self.rng.shuffle(idxs)
            # else:
            #     pass

            for idx in idxs:
                img_meta = coco.imgs[keys[idx]]
                img_idx = img_meta['id']
                ann_idx = coco.getAnnIds(imgIds=img_idx)

                if 'http://' in self.img_path:
                    img_url = self.img_path + img_meta['file_name']
                else:
                    img_url = os.path.join(self.img_path, img_meta['file_name'])

                anns = coco.loadAnns(ann_idx)
                # miss_mask = np.zeros((img_meta['height'], img_meta['width'])).astype(np.bool)
                # for ann in anns:
                #     if ann['num_keypoints'] <= 0:
                #         tmp_mask = coco.annToMask(ann)
                #         # plt.imshow(tmp_mask.astype(np.int))
                #         miss_mask = np.logical_or(miss_mask, tmp_mask)
                #         # plt.imshow(miss_mask.astype(np.int))
                #         # plt.show()
                meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

                # total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
                # if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                #     continue
                metas.append(meta)
            yield metas


class ParkinsonValDataLoader():
    def __init__(self, path, tar_w=600, img_path=None, is_train=False, decode_img=True, only_idx=-1,
                 image_num=9, result_save_path='result/json/ill_val.json'):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.image_num = image_num
        self.target_w = tar_w
        self.target_h = None
        self.ori_h = None
        self.ori_w = None
        self.imgs_info = []
        self.annos_info = []
        self.result_save_path = result_save_path
        if is_train:
            whole_path = path
        else:
            whole_path = path
        # self.img_path = (img_path if img_path is not None else '') + ('train2017/' if is_train else 'val2017/')
        self.img_path = img_path
        self.coco = COCO(whole_path)
        logger.info('%s dataset %d' % (path, self.size()))

    def get_coco_val_anno(self, keypoints, image_id):
        annotations = []
        file_name = str(image_id).rjust(8, '0') + '.jpg'
        image = {"file_name": file_name, "id": image_id}
        for i, keypoint in enumerate(keypoints):
            annotations.append(
                {"num_keypoints": 17, "keypoints": keypoint, "image_id": image_id, "id": int(image_id * 10 + i)})
        return image, annotations

    def get_last_keypoints_from_bodies(self, bodys):
        if len(bodys) == 0:
            keypoint = [0 for _ in range(51)]
        else:
            for human in bodys:
                keypoint = []
                for coco_part in cross_keypoints_list:
                    if coco_part is not None:
                        if coco_part.value not in human.body_parts.keys():
                            keypoint = keypoint + [0, 0, 0]
                            continue
                        keypoint = keypoint + [int(human.body_parts[coco_part.value].x * self.ori_w + 0.5),
                                               int(human.body_parts[coco_part.value].y * self.ori_h + 0.5), 1]
                    else:
                        keypoint = keypoint + [0, 0, 0]
                        continue
        return keypoint

    def size(self):
        return len(self.coco.imgs)

    def get_data(self):
        idxs = np.arange(self.size())
        # idxs=range(100)
        keys = list(self.coco.imgs.keys())
        for idx in idxs:
            img_meta = self.coco.imgs[keys[idx]]
            img_id = img_meta['id']
            if img_id in parkinson_cut_list:
                file_name = img_meta['file_name']
                img_url = os.path.join(self.img_path, file_name)
                image, self.ori_h, self.ori_w = read_image_path(img_url)
                self.target_h = int(self.target_w * (self.ori_h / self.ori_w))
                print(self.ori_h, self.ori_w)
                image = np.array(cv2.resize(image, (self.target_w, self.target_h)))
                images = np.repeat(image[np.newaxis, np.newaxis, :], self.image_num, axis=1)
            else:
                images = np.delete(images, 0, axis=1)
                file_name = img_meta['file_name']
                img_url = os.path.join(self.img_path, file_name)
                image, _, _ = read_image_path(img_url)
                image = np.array(cv2.resize(image, (self.target_w, self.target_h)))
                images = np.concatenate((images, image[np.newaxis, np.newaxis, :]), axis=1)
            yield images, img_id

    def add_result(self, bodys, image_id):
        keypoints = []
        keypoints.append(self.get_last_keypoints_from_bodies(bodys))
        img_info, anno_info = self.get_coco_val_anno(keypoints=keypoints, image_id=image_id)
        self.imgs_info.append(img_info)
        self.annos_info += anno_info

    def caculate_accuracy(self):
        ref = {"images": self.imgs_info, "annotations": self.annos_info}
        with open(self.result_save_path, "w") as f:
            json.dump(ref, f)
            print('writed to ' + self.result_save_path)
        smooth_path = smoother(self.result_save_path)
        pre_coco = COCO(smooth_path)
        idxs = np.arange(self.size())
        # idxs = range(100)
        keys_gt = list(self.coco.imgs.keys())
        keys_pre = list(pre_coco.imgs.keys())
        vec_error_list = []
        dis_error_list = []
        for idx in idxs:
            img_meta_gt = self.coco.imgs[keys_gt[idx]]
            img_idx_gt = img_meta_gt['id']
            ann_idx_gt = self.coco.getAnnIds(imgIds=img_idx_gt)
            keypoints_gt = self.coco.loadAnns(ann_idx_gt)[0]
            kp_gt = np.array(keypoints_gt['keypoints'])
            kp_gt.resize([17, 3])

            img_meta_pre = pre_coco.imgs[keys_pre[idx]]
            img_idx_pre = img_meta_pre['id']
            ann_idx_pre = pre_coco.getAnnIds(imgIds=img_idx_pre)
            keypoints_pre = pre_coco.loadAnns(ann_idx_pre)[0]
            kp_pre = np.array(keypoints_pre['keypoints'])
            kp_pre.resize([17, 3])

            if img_idx_gt in parkinson_cut_list or img_idx_pre in parkinson_cut_list:
                kp_val_gt = np.repeat(kp_gt[np.newaxis, :, :], 2, 0)
                kp_val_pre = np.repeat(kp_pre[np.newaxis, :, :], 2, 0)
            else:
                kp_val_gt = np.delete(kp_val_gt, 0, axis=0)
                kp_val_gt = np.concatenate((kp_val_gt, kp_gt[np.newaxis, :, :]), axis=0)
                kp_val_pre = np.delete(kp_val_pre, 0, axis=0)
                kp_val_pre = np.concatenate((kp_val_pre, kp_pre[np.newaxis, :, :]), axis=0)

            for i in range(kp_gt.shape[0]):
                if kp_val_gt[0, i, 2] == 0 or kp_val_gt[1, i, 2] == 0 or kp_val_pre[0, i, 2] == 0 or kp_val_pre[
                    1, i, 2] == 0:
                    continue
                vec_gt = kp_val_gt[1, i, :-1] - kp_val_gt[0, i, :-1]
                vec_pre = kp_val_pre[1, i, :-1] - kp_val_pre[0, i, :-1]
                vec_error = np.linalg.norm(vec_gt - vec_pre)
                vec_error_list.append(vec_error)

            for i in range(kp_gt.shape[0]):
                if kp_val_gt[1, i, 2] == 0 or kp_val_pre[1, i, 2] == 0:
                    continue
                dis_error = np.linalg.norm(kp_val_gt[1, i, :-1] - kp_val_pre[1, i, :-1])
                dis_error_list.append(dis_error)
        average_vec_error = np.average(np.array(vec_error_list))
        vec_var = np.var(np.array(vec_error_list))

        average_dis_error = np.average(np.array(dis_error_list))
        dis_var = np.var(np.array(dis_error_list))
        return average_dis_error, average_vec_error, dis_var, vec_var


class ParkinsonPose(RNGDataFlow):
    def __init__(self, path, img_path=None, is_train=True, decode_img=True, only_idx=-1, image_num=5):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.image_num = image_num
        if is_train:
            whole_path = path
        else:
            whole_path = os.path.join(path, 'person_keypoints_val2017.json')
        # self.img_path = (img_path if img_path is not None else '') + ('train2017/' if is_train else 'val2017/')
        self.img_path = img_path
        self.coco = COCO(whole_path)
        logger.info('%s dataset %d' % (path, self.size()))

    def size(self):
        return len(self.coco.imgs)

    def notin_cut_list(self, idx):
        keys = list(self.coco.imgs.keys())
        if idx < len(keys) - self.image_num:
            for i in range(self.image_num):
                img_meta = self.coco.imgs[keys[idx + i]]
                img_id = img_meta['id']
                if img_id in parkinson_cut_list:
                    return False
            return True
        else:
            return False

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
            idxs = filter(self.notin_cut_list, idxs)
            keys = list(self.coco.imgs.keys())
            for begin_idx in idxs:
                metas = []
                for idx in range(begin_idx, begin_idx + self.image_num):
                    img_meta = self.coco.imgs[keys[idx]]
                    img_idx = img_meta['id']
                    ann_idx = self.coco.getAnnIds(imgIds=img_idx)

                    if 'http://' in self.img_path:
                        img_url = self.img_path + img_meta['file_name']
                    else:
                        img_url = os.path.join(self.img_path, img_meta['file_name'])

                    anns = self.coco.loadAnns(ann_idx)
                    meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

                    # total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
                    # if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                    #     continue
                    metas.append(meta)
                yield metas
        else:
            idxs = filter(self.notin_cut_list, idxs)
            keys = list(self.coco.imgs.keys())
            for begin_idx in idxs:
                metas = []
                for idx in range(begin_idx, begin_idx + self.image_num):
                    img_meta = self.coco.imgs[keys[idx]]
                    img_idx = img_meta['id']
                    ann_idx = self.coco.getAnnIds(imgIds=img_idx)

                    if 'http://' in self.img_path:
                        img_url = self.img_path + img_meta['file_name']
                    else:
                        img_url = os.path.join(self.img_path, img_meta['file_name'])

                    anns = self.coco.loadAnns(ann_idx)
                    meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

                    # total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
                    # if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                    #     continue
                    metas.append(meta)
                yield metas


class CocoPose(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        global mplset
        # if as_numpy and not mplset:
        #     import matplotlib as mpl
        #     mpl.use('Agg')
        mplset = True

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
            whole_path = path
        else:
            whole_path = os.path.join(path, 'person_keypoints_val2017.json')
        # self.img_path = (img_path if img_path is not None else '') + ('train2017/' if is_train else 'val2017/')
        self.img_path = img_path
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
            if not img_meta['is_labeled']:
                continue
            img_idx = img_meta['id']
            ann_idx = self.coco.getAnnIds(imgIds=img_idx)

            if 'http://' in self.img_path:
                img_url = self.img_path + img_meta['file_name']
            else:
                img_url = os.path.join(self.img_path, img_meta['file_name'])

            anns = self.coco.loadAnns(ann_idx)
            # miss_mask = np.zeros((img_meta['height'], img_meta['width'])).astype(np.bool)
            # for ann in anns:
            #     if ann['num_keypoints'] <= 0:
            #         tmp_mask = self.coco.annToMask(ann)
            #         # plt.imshow(tmp_mask.astype(np.int))
            #         miss_mask = np.logical_or(miss_mask, tmp_mask)
            #         # plt.imshow(miss_mask.astype(np.int))
            #         # plt.show()
            meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

            # total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
            # if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
            #     continue

            yield [meta]


class MPIIPose(RNGDataFlow):
    def __init__(self):
        pass

    def size(self):
        pass

    def get_data(self):
        pass


def read_image_url(metas):
    for meta in metas:
        img_str = None
        if 'http://' in meta.img_url:
            # print(meta.img_url)
            for _ in range(10):
                try:
                    resp = requests.get(meta.img_url)
                    if resp.status_code // 100 != 2:
                        logger.warning('request failed code=%d url=%s' % (resp.status_code, meta.img_url))
                        time.sleep(1.0)
                        continue
                    img_str = resp.content
                    break
                except Exception as e:
                    logger.warning('request failed url=%s, err=%s' % (meta.img_url, str(e)))
        else:
            img_str = open(meta.img_url, 'rb').read()

        if not img_str:
            logger.warning('image not read, path=%s' % meta.img_url)
            raise Exception()

        nparr = np.fromstring(img_str, np.uint8)
        meta.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        meta.height = meta.img.shape[0]
        meta.width = meta.img.shape[1]
    return metas


def get_parkinson_dataflow(path, is_train, img_path=None):
    ds = ParkinsonPose(path, img_path, is_train)  # read data from lmdb
    if is_train:
        ds = MapData(ds, read_image_url)
        ds = MapData(ds, pose_random_scale)
        ds = MapData(ds, pose_rotation)
        # ds = MapDataComponent(ds, pose_flip)
        ds = MapData(ds, pose_resize_shortestedge_random)
        ds = MapData(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        # augs = [
        #     imgaug.RandomApplyAug(imgaug.RandomChooseAug([
        #         imgaug.GaussianBlur(max_size=3)
        #     ]), 0.7)
        # ]
        # ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() - 1)
    else:
        ds = MultiThreadMapData(ds, nr_thread=8, map_func=read_image_url, buffer_size=1000)
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)

    return ds


def get_dataflow(path, is_train, img_path=None):
    ds = PoseTrackPose(path, img_path, is_train)  # read data from lmdb
    if is_train:

        ds = MapData(ds, read_image_url)
        ds = MapData(ds, pose_gen_mask)
        ds = MapData(ds, pose_random_scale)
        ds = MapData(ds, pose_rotation)
        # # ds = MapDataComponent(ds, pose_flip)
        ds = MapData(ds, pose_resize_shortestedge_random)
        ds = MapData(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        # augs = [
        #     imgaug.RandomApplyAug(imgaug.RandomChooseAug([
        #         imgaug.GaussianBlur(max_size=3)
        #     ]), 0.7)
        # ]
        # ds = AugmentImageComponent(ds, augs)
        # ds = PrefetchData(ds, 100, multiprocessing.cpu_count())
        ds = PrefetchDataZMQ(ds, multiprocessing.cpu_count())

    else:
        ds = MultiThreadMapData(ds, nr_thread=8, map_func=read_image_url, buffer_size=1000)
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)

    return ds


def _get_dataflow_onlyread(path, is_train, img_path=None):
    ds = CocoPose(path, img_path, is_train)  # read data from lmdb
    ds = MapData(ds, read_image_url)
    ds = MapData(ds, pose_to_img)
    # ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() * 4)
    return ds


def get_dataflow_batch(path, is_train, batchsize, img_path=None):
    logger.info('dataflow img_path=%s' % img_path)
    ds = get_dataflow(path, is_train, img_path=img_path)
    ds = BatchData(ds, batchsize)
    if is_train:
        # ds = PrefetchData(ds, 10, 2)
        ds = PrefetchDataZMQ(ds, 2)
    else:
        ds = PrefetchData(ds, 50, 2)

    return ds


def get_parkinson_dataflow_batch(path, is_train, batchsize, img_path=None):
    logger.info('dataflow img_path=%s' % img_path)
    ds = get_parkinson_dataflow(path, is_train, img_path=img_path)
    ds = BatchData(ds, batchsize)
    if is_train:
        ds = PrefetchData(ds, 10, 2)
    else:
        ds = PrefetchData(ds, 50, 2)

    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders],
                                  shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.__flag = threading.Event()
        self.__flag.set()

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                self.__flag.wait()
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    set_network_input_wh(368, 368)
    set_network_input_wh(640, 360)
    set_network_scale(1)

    df = get_dataflow('/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/'
                      'data_select/train/json_data/', True,
                      '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/'
                      'data_select/train/')

    # df = get_dataflow('/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/'
    #                   'posetrack_dataset/posetrack_data/annotations/train/', True,
    #                   '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/posetrack_dataset'
    #                   '/posetrack_data/')
    # df = get_parkinson_dataflow('/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/dataset/annotations/'
    #                             'person_keypoints_train2017.json', True,
    #                             '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/dataset/images/train2017/')
    # df = get_dataflow('/Volumes/data-1/yzy/dynamic/posetrack_dataset/posetrack_data/annotations/train/', True,
    #                   '/Volumes/data-1/yzy/dynamic/posetrack_dataset/posetrack_data/images/train/000001_bonn_train/')
    # df = _get_dataflow_onlyread('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/')
    # df = get_dataflow('/root/coco/annotations', False, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')

    # from tensorpack.dataflow.common import TestDataSpeed

    # TestDataSpeed(df).start()
    # sys.exit(0)

    with tf.Session() as sess:
        df.reset_state()
        # t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logger.info('%d dp shape={}'.format(d.shape))
            # print(time.time() - t1)
            # t1 = time.time()
            for i in range(5):
                CocoPose.display_image(dp[0][i, :, :, :], dp[1][i, :, :, :].astype(np.float32),
                                       dp[2][i, :, :, :].astype(np.float32))
            print(dp[1].shape, dp[2].shape)
    #
    #
    # logger.info('done')
    # # global _network_w, _network_h, _scale
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # # set_network_input_wh(368, 368)
    # set_network_input_wh(640, 360)
    # set_network_scale(8)
    # anno_tarin_path = '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/posetrack_dataset/posetrack_data/annotations/train/'
    # annos = os.listdir(anno_tarin_path)
    # writer = tfrecods_writer('dataset/test.tfrecords')
    # for anno in tqdm(annos):
    #     if anno.split('.')[-1] != 'json':
    #         continue
    #     df = get_dataflow(anno_tarin_path + anno, True, '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dynamic/posetrack_dataset/posetrack_data/')
    #     imgs = np.zeros((1, _network_h, _network_w, 3), dtype=np.float16)
    #     heatmaps = np.zeros((1, _network_h // _scale, _network_w // _scale, 16), dtype=np.float16)
    #     pafs = np.zeros((1, _network_h // _scale, _network_w // _scale, 30), dtype=np.float16)
    #     img_num = 0
    #     with tf.Session() as sess:
    #         df.reset_state()
    #         for idx, dp in enumerate(df.get_data()):
    #             if idx == 0:
    #                 for d in dp:
    #                     logger.info('%d dp shape={}'.format(d.shape))
    #             # CocoPose.display_image(dp[0], dp[1].astype(np.float32), dp[2].astype(np.float32))
    #             if img_num != 0:
    #                 imgs = np.concatenate((imgs, dp[0][np.newaxis, :, :, :]), axis=0)
    #                 heatmaps = np.concatenate((heatmaps, dp[1][np.newaxis, :, :, :]), axis=0)
    #                 pafs = np.concatenate((pafs, dp[2][np.newaxis, :, :, :]), axis=0)
    #             else:
    #                 imgs = dp[0][np.newaxis, :, :, :]
    #                 heatmaps = dp[1][np.newaxis, :, :, :]
    #                 pafs = dp[2][np.newaxis, :, :, :]
    #             img_num += 1
    #             print(dp[0].shape, dp[1].shape, dp[2].shape, img_num)
    #             # if img_num == 2:
    #             #     CocoPose.display_image(imgs[1, :, :, :], heatmaps[1, :, :, :].astype(np.float32),
    #             #                            pafs[1, :, :, :].astype(np.float32))
    #             #     break
    #         writer.new_example(img_num=np.array(img_num, dtype=np.float16), imgs=imgs, heatmaps=heatmaps, pafs=pafs)
    # writer.finishi_write()
    # df = get_dataflow('/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
    #                   'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/', True, '/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
    #                              'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/')

    # df = _get_dataflow_onlyread('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/')
    # df = get_dataflow('/root/coco/annotations', False, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')

    # from tensorpack.dataflow.common import TestDataSpeed
    # TestDataSpeed(df).start()
    # sys.exit(0)

    # logger.info('done')

# import tensorflow as tf
# from pose_dataset import CocoPose
# from tqdm import tqdm
#
#
#
# class tfrecods_writer():
#     def __init__(self, train_filename):
#         self.writer = tf.python_io.TFRecordWriter(train_filename)
#
#     def _bytes_feature(self, value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#     def new_example(self, idx, joint_list, img_meta, anns, miss_mask):
#         feature = {
#             'idx': self._bytes_feature(tf.compat.as_bytes(idx.tostring())),
#             'img_meta': self._bytes_feature(tf.compat.as_bytes(img_meta.tostring())),
#             'anns': self._bytes_feature(tf.compat.as_bytes(anns.tostring())),
#             'miss_mask': self._bytes_feature(tf.compat.as_bytes(miss_mask.tostring())),
#             'joint_list': self._bytes_feature(tf.compat.as_bytes(joint_list))
#         }
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#         self.writer.write(example.SerializeToString())
#
#     def finishi_write(self):
#         self.writer.close()
#
#
# if __name__ == '__main__':
#     ds = CocoPose('/Volumes/data-1/yzy/dataset/'
#                           'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/', '/Volumes/data-1/yzy/dataset/'
#                           'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/', False)
#     writer = tfrecods_writer('train.tfrecords')
#
#     for meta in tqdm(ds.get_data()):
#         idx = meta[0].idx
#         img_url = meta[0].img_url
#         miss_mask = meta[0].miss_mask
#         height = meta[0].height
#         width = meta[0].width
#         joint_list = meta[0].joint_list
#         meta_ = meta[0].meta
#         anns = meta[0].annotations
#         writer.new_example(idx=idx, joint_list=joint_list, img_meta=meta_, anns=anns, miss_mask=miss_mask)
# smoother('result/PD_data/dispraed/baseline/301/UE_P_hand_coco.json')
# smoother('result/json/deep_high_res.json')
