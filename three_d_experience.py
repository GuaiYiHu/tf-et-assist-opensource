import copy
import json
import os

import cv2
import numpy as np
import yaml
from cv2 import aruco
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_order(name):
    if name.split('.')[-1] == 'jpg':
        element = name.split('-')[-1]
        element = element.split('.')[0]
        return int(element)
    else:
        return 999999


def find_peak(points, marg=5):
    vals = []
    ids = []
    for i in range(len(points) - 2 * marg):
        if False in list(points[i: i + 2 * marg] <= points[i + marg]):
            pass
        else:
            vals.append(points[i + marg])
            ids.append(i + marg)
    return vals, ids


def find_valley(points, marg=5):
    vals = []
    ids = []
    for i in range(len(points) - 2 * marg):
        if False in list(points[i: i + 2 * marg] >= points[i + marg]):
            pass
        else:
            vals.append(points[i + marg])
            ids.append(i + marg)
    return vals, ids


def smooth(points):
    pass


def read_json(path):
    data = json.load(open(path, 'r'))
    positions = data['position']
    return positions


def read_txt(path):
    data = []
    for line in open(path, "r"):
        x, y, z = line.split(' ')
        data.append([float(x), float(y), float(z)])
    return data


def calculate_frequence():
    pass


def inter(position):
    ids = []
    peaks = []
    inter_position = []
    for i, p in enumerate(position):
        if p != [None, None]:
            ids.append(i)
            peaks.append(p)

    xvals = np.linspace(0, len(position) - 1, len(position))
    print(len(np.array(peaks)[:, 0]))
    x_peaks_inter = np.interp(xvals, ids, np.array(peaks)[:, 0])
    y_peaks_inter = np.interp(xvals, ids, np.array(peaks)[:, 1])
    for x, y in zip(x_peaks_inter, y_peaks_inter):
        inter_position.append([x, y])
    plt.plot(np.array(inter_position)[:, 0])
    plt.plot(np.array(position)[:, 0])
    # plt.plot(peaks_inter)
    plt.show()

    return inter_position


def calculate_amplitude(sf):
    peaks, p_ids = find_peak(sf)
    valleys, v_ids = find_valley(sf)
    xvals = np.linspace(0, sf.shape[0], sf.shape[0])
    peaks_inter = np.interp(xvals, p_ids, peaks)
    valleys_inter = np.interp(xvals, v_ids, valleys)
    a_inter = (peaks_inter - valleys_inter)
    return a_inter, peaks_inter, valleys_inter


def detect_marker(father_path, exp_list):
    camera_list = ['cam1', 'cam2']
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParams = aruco.DetectorParameters_create()

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    for exp_num in exp_list:
        print(exp_num)
        for camera_name in camera_list:
            saver_initialized = False
            path = father_path + exp_num + '/' + camera_name + '/'

            pic_list = os.listdir(path)
            pic_list.sort(key=get_order)

            position = []
            for img_name in tqdm(pic_list):
                if img_name.split('.')[-1] == 'jpg':

                    raw_img = cv2.imread(path + '/' + img_name)
                    img = copy.deepcopy(raw_img)
                    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    if not saver_initialized:
                        h, w = im_gray.shape[:2]
                        video_saver = cv2.VideoWriter(path + camera_name + '.mp4', fourcc, 30, (w, h))
                        marker_video_saver = cv2.VideoWriter(path + camera_name + '_marker.mp4', fourcc, 30, (w, h))
                        saver_initialized = True
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
                    if len(corners) == 0:
                        position.append([None, None])
                        print('drop!')
                    else:
                        for corner in corners[:1]:
                            cv2.circle(img, (corner[0][2][0], corner[0][2][1]), 5, (0, 255, 0))
                            position.append([int(corner[0][2][0]), int(corner[0][2][1])])
                    marker_video_saver.write(cv2.resize(img, (w, h)))
                    video_saver.write(cv2.resize(raw_img, (w, h)))
            position = inter(position)
            ref = {"position": position}
            with open(path + camera_name + '.json', "w") as f:
                json.dump(ref, f)

