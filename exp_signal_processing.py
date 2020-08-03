import argparse
import json
import os

import numpy as np
import pandas as pd
import pywt
import scipy as sc
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from scipy import signal
from sklearn import metrics

import dis_anno
import get_pic_from_video
import get_video_list
import run_hand
import run_lstm
from data_analyze import getPoint
from three_d_experience import calculate_amplitude, read_txt

font = FontProperties(fname=r"simsun.ttc", size=10)


def map2right(result, right_index):
    return list(map(result.__getitem__, right_index))


def caculate_frequency(points, fps=30):
    cwtmatr1, freqs1 = pywt.cwt(points, np.arange(1, 100), 'cgau8', 1 / fps)
    # cwtmatr1= pywt.dwt(sf[:i + fps, keypoint_index], 'db2')

    cwt_frequency_list = [np.inf]
    cwt_a_list = []
    for j in range(cwtmatr1.shape[1]):
        cwt_abs = np.abs(cwtmatr1[:, j])
        max_cwt_abs = np.max(cwt_abs)
        try:
            cwt_f_index = np.where(cwt_abs == max_cwt_abs)[0][0]
            cwt_f = freqs1[cwt_f_index]
        except:
            cwt_f = 5
        cwt_frequency_list.append(cwt_f)
        cwt_a_list.append(max_cwt_abs)
    return cwt_frequency_list


def caculate_max_amplitude(amplitude, frequency_list, maintain_time_threshold=0.3, fps=30):
    maintain_frame = int(maintain_time_threshold * fps)
    windows_num = amplitude.shape[0] - maintain_frame
    slid_windows = np.zeros(shape=[windows_num, maintain_frame])
    for i in range(windows_num):
        slid_windows[i, :] = amplitude[i:i + maintain_frame]
    min_in_window = np.nanmin(slid_windows, axis=1)
    for i, a in enumerate(list(min_in_window)):
        if frequency_list[i] <= 3 or frequency_list[i] >= 12:
            min_in_window[i] = 0
    maintain_amplitude = np.max(min_in_window)
    return maintain_amplitude


def caculate_amplitude(sf):
    peaks, p_ids = find_peak(sf)
    valleys, v_ids = find_valley(sf)
    xvals = np.linspace(0, sf.shape[0], sf.shape[0])
    peaks_inter = np.interp(xvals, p_ids, peaks)
    valleys_inter = np.interp(xvals, v_ids, valleys)
    a_inter = (peaks_inter - valleys_inter)
    return a_inter, [peaks, p_ids], [valleys, v_ids], peaks_inter, valleys_inter


def inter_miss_point(points):
    for idx, num in enumerate(points):
        if num == 0:
            n = idx
            try:
                while points[n] == 0:
                    n += 1
                if idx == 0:
                    points[idx] = points[n]
                else:
                    points[idx] = int((points[idx - 1] + points[n]) / 2 + 0.5)
            except:
                points[idx] = points[idx - 1]
    return points


def find_peak(points, marg=5):
    vals = []
    ids = []
    for i in range(len(points) - 2 * marg):
        # p = points[i: i + 2 * marg] >= points[i + marg]
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
        # p = points[i: i + 2 * marg] >= points[i + marg]
        if False in list(points[i: i + 2 * marg] >= points[i + marg]):
            pass
        else:
            vals.append(points[i + marg])
            ids.append(i + marg)
    return vals, ids
    # for j in range(2*marg):
    #     if points[i+j] < points[i+marg]:
    #     points[i+marg]


def find_wave_start_stop(points, threshold, f_number=10):
    start_ids = []
    stop_ids = []
    state = 0

    for i in range(len(points) - f_number):
        if state == 0:
            if False in list(points[i:i + f_number] >= threshold):
                pass
            else:
                state = 1
                start_ids.append(i)
        else:
            if False in list(points[i:i + f_number] < threshold):
                pass
            else:
                state = 0
                stop_ids.append(i)
    return start_ids, stop_ids


def multiVector(A, B):
    C = sc.zeros(len(A))

    for i in range(len(A)):
        C[i] = A[i] * B[i]

    return sum(C)


def inVector(A, b, a):
    D = sc.zeros(b - a + 1)

    for i in range(b - a + 1):
        D[i] = A[i + a]

    return D[::-1]


def LMS(xn, dn, M, mu, itr):
    en = sc.zeros(itr)

    W = [[0] * M for i in range(itr)]

    for k in range(itr)[M - 1:itr]:
        x = inVector(xn, k, k - M + 1)
        d = x.mean()

        y = multiVector(W[k - 1], x)

        en[k] = d - y

        W[k] = np.add(W[k - 1], 2 * mu * en[k] * x)  # 跟新权重

    # 求最优时滤波器的输出序列

    yn = sc.inf * sc.ones(len(xn))

    for k in range(len(xn))[M - 1:len(xn)]:
        x = inVector(xn, k, k - M + 1)

        yn[k] = multiVector(W[len(W) - 1], x)

    return (yn, en)


def smoother(path):
    data = json.load(open(path, 'r'))
    real_shoulder_distance = 40
    begin = 0
    finish = 9700
    points = []
    for frame in data['annotations']:
        points.append(frame['keypoints'])
    l_wrist_x = np.array(points)[begin:finish, 27]
    l_wrist_y = np.array(points)[begin:finish, 28]
    r_wrist_x = np.array(points)[begin:finish, 30]
    r_wrist_y = np.array(points)[begin:finish, 31]
    l_shoulder = np.array(points)[begin:finish, 15:17]
    r_shoulder = np.array(points)[begin:finish, 18:20]
    l_r_shoulder_distance = np.mean(np.sqrt(np.sum(np.square(l_shoulder - r_shoulder), axis=1)))

    l_wrist_x = inter_miss_point(l_wrist_x)
    l_wrist_y = inter_miss_point(l_wrist_y)
    r_wrist_x = inter_miss_point(r_wrist_x)
    r_wrist_y = inter_miss_point(r_wrist_y)

    b, a = signal.butter(1, [0.2, 0.6], 'bandpass')
    l_wrist_x = signal.filtfilt(b, a, l_wrist_x, axis=0)
    l_wrist_y = signal.filtfilt(b, a, l_wrist_y, axis=0)
    r_wrist_x = signal.filtfilt(b, a, r_wrist_x, axis=0)
    r_wrist_y = signal.filtfilt(b, a, r_wrist_y, axis=0)

    l_wrist_x_amplitude, l_wrist_x_peaks, l_wrist_x_valleys, l_wrist_x_up, l_wrist_x_down = caculate_amplitude(
        l_wrist_x)
    l_wrist_y_amplitude, l_wrist_y_peaks, l_wrist_y_valleys, l_wrist_y_up, l_wrist_y_down = caculate_amplitude(
        l_wrist_y)
    r_wrist_x_amplitude, r_wrist_x_peaks, r_wrist_x_valleys, l_wrist_x_up, l_wrist_x_down = caculate_amplitude(
        r_wrist_x)
    r_wrist_y_amplitude, r_wrist_y_peaks, r_wrist_y_valleys, r_wrist_y_up, r_wrist_y_down = caculate_amplitude(
        r_wrist_y)
    l_wrist_amplitude = np.sqrt(np.square(l_wrist_x_amplitude * 1.7) + np.square(l_wrist_y_amplitude))
    r_wrist_amplitude = np.sqrt(np.square(r_wrist_x_amplitude * 1.7) + np.square(r_wrist_y_amplitude))
    l_wrist_y_frenquency = caculate_frequency(l_wrist_y)
    r_wrist_y_frenquency = caculate_frequency(r_wrist_y)

    l_wrist_maintain_amplitude = caculate_max_amplitude(l_wrist_amplitude,
                                                        l_wrist_y_frenquency) / l_r_shoulder_distance * real_shoulder_distance
    r_wrist_maintain_amplitude = caculate_max_amplitude(r_wrist_amplitude,
                                                        r_wrist_y_frenquency) / l_r_shoulder_distance * real_shoulder_distance
    positions = read_txt('data/' + path.split('/')[
        -2] + '/shake.txt')[:-9]

    x_positions = np.array([p[0] * 100 for p in positions])[begin:finish]
    y_positions = np.array([p[1] * 100 for p in positions])[begin:finish]
    z_positions = np.array([p[2] * 100 for p in positions])[begin:finish]
    x_positions = signal.filtfilt(b, a, x_positions, axis=0)
    y_positions = signal.filtfilt(b, a, y_positions, axis=0)
    z_positions = signal.filtfilt(b, a, z_positions, axis=0)
    x_amplitude, x_peaks, x_valleys = calculate_amplitude(x_positions)
    y_amplitude, y_peaks, y_valleys = calculate_amplitude(y_positions)
    z_amplitude, z_peaks, z_valleys = calculate_amplitude(z_positions)
    amplitude = np.sqrt(np.square(x_amplitude) + np.square(y_amplitude) + np.square(z_amplitude))
    r_wrist_y_frenquency_gt = caculate_frequency(y_positions)
    r_wrist_maintain_amplitude_gt = caculate_max_amplitude(amplitude,
                                                           r_wrist_y_frenquency_gt)
    return getPoint(r_wrist_maintain_amplitude_gt), getPoint(
        r_wrist_maintain_amplitude), r_wrist_maintain_amplitude, r_wrist_maintain_amplitude_gt


def caculate_hand_motion(dir):
    fps = 30
    gs = GridSpec(12, 1)
    json_data_list = os.listdir(dir)
    left_hand_distances = []
    right_hand_distances = []
    for json_data_name in json_data_list:
        data = json.load(open(dir + json_data_name, 'r'))
        left_hand = data['people'][0]['hand_left_keypoints_2d']
        right_hand = data['people'][0]['hand_right_keypoints_2d']
        left_hand_x = left_hand[0::3]
        left_hand_y = left_hand[1::3]
        left_hand_s = left_hand[2::3]
        right_hand_x = right_hand[0::3]
        right_hand_y = right_hand[1::3]
        right_hand_s = right_hand[2::3]
        left_distance = np.sqrt(
            np.sum(np.square(np.array([left_hand_x[4], left_hand_y[4]]) - np.array([left_hand_x[8], left_hand_y[8]]))))
        right_distance = np.sqrt(
            np.sum(
                np.square(np.array([right_hand_x[4], right_hand_y[4]]) - np.array([right_hand_x[8], right_hand_y[8]]))))
        left_hand_distances.append(left_distance)
        right_hand_distances.append(right_distance)
    b, a = signal.butter(1, [0.1, 0.3], 'bandpass')
    left_hand_distances = signal.filtfilt(b, a, left_hand_distances, axis=0)
    right_hand_distances = signal.filtfilt(b, a, right_hand_distances, axis=0)

    plt.ion()
    plt.figure(figsize=(19.2, 21.6), dpi=50)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.96, top=0.93,
                        wspace=0.13, hspace=0.33)
    for i in range(left_hand_distances.shape[0] - fps):
        plt.clf()
        try:
            cwtmatr1, freqs1 = pywt.cwt(left_hand_distances[:i + fps], np.arange(1, 100), 'cgau8', 1 / fps)

            cwt_frequency_list_left = [np.inf]
            cwt_a_list = []
            for j in range(len(cwtmatr1[0, :i + fps])):
                cwt_abs = np.abs(cwtmatr1[:, j])
                max_cwt_abs = np.max(cwt_abs)
                try:
                    cwt_f_index = np.where(cwt_abs == max_cwt_abs)[0][0]
                    cwt_f = freqs1[cwt_f_index]
                except:
                    cwt_f = 5
                cwt_frequency_list_left.append(cwt_f)
                cwt_a_list.append(max_cwt_abs)
        except:
            pass

        try:
            cwtmatr2, freqs2 = pywt.cwt(right_hand_distances[:i + fps], np.arange(1, 100), 'cgau8', 1 / fps)

            cwt_frequency_list_right = [np.inf]
            cwt_a_list = []
            for j in range(len(cwtmatr2[0, :i + fps])):
                cwt_abs = np.abs(cwtmatr2[:, j])
                max_cwt_abs = np.max(cwt_abs)
                try:
                    cwt_f_index = np.where(cwt_abs == max_cwt_abs)[0][0]
                    cwt_f = freqs2[cwt_f_index]
                except:
                    cwt_f = 5
                cwt_frequency_list_right.append(cwt_f)
                cwt_a_list.append(max_cwt_abs)
        except:
            pass

        agraphic = plt.subplot(gs[0:2, 0])
        agraphic.set_title('左手', fontproperties=font)
        agraphic.plot(left_hand_distances[0:i + fps])

        bgraphic = plt.subplot(gs[2:4, 0])
        bgraphic.set_title('右手', fontproperties=font)
        bgraphic.plot(right_hand_distances[0:i + fps])

        hgraphic = plt.subplot(gs[4:6, 0])
        hgraphic.set_title('左手频率曲线', fontproperties=font)
        hgraphic.plot(cwt_frequency_list_left)

        hgraphic = plt.subplot(gs[6:8, 0])
        hgraphic.set_title('右手频率曲线', fontproperties=font)
        hgraphic.plot(cwt_frequency_list_right)
        plt.pause(0.01)
        plt.savefig('result/PD_data/dispraed/baseline/307/result/' + str(i) + '.jpg')
    plt.ioff()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test codes for Openpose using Tensorflow')
    parser.add_argument('--mode', '-m', help='Test mode, "cal" or test', default="cal")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/fine_tone_doctor/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--pd_video', type=str, default='images/PD_data/selected/307_1_month.mp4')
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--stage_num', type=int, default=6)
    parser.add_argument('--hm_channels', type=str, default=15)
    parser.add_argument('--paf_channels', type=str, default=26)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--save_json_path', type=str, default='images/PD_data/selected/result/307_1_month_s.json')
    parser.add_argument('--val_anno_path', type=str, default=None)
    parser.add_argument('--val_img_path', type=str, default='/run/user/1000/gvfs/smb-share:server=server,share=data'
                                                            '/yzy/dynamic/dataset/images/val2017/')
    parser.add_argument('--hand_use_bn', type=bool, default=False)
    parser.add_argument('--hand_train_vgg', type=bool, default=False)
    parser.add_argument('--hand_checkpoint_path', type=str, default='checkpoints/train/2019-11-15-12-13-19/')
    parser.add_argument('--hand_backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    args = parser.parse_args()

    mode = args.mode
    checkpoint_path = args.checkpoint_path
    stage_num = args.stage_num
    hm_channels = args.hm_channels
    paf_channels = args.paf_channels
    use_bn = args.use_bn
    train_vgg = args.train_vgg
    backbone_net_ckpt_path = args.backbone_net_ckpt_path
    val_anno_path = args.val_anno_path
    val_img_path = args.val_img_path
    save_json_path = args.save_json_path
    pd_video = args.pd_video
    hand_use_bn = args.hand_use_bn
    hand_train_vgg = args.hand_train_vgg
    hand_checkpoint_path = args.hand_checkpoint_path
    hand_backbone_net_ckpt_path = args.hand_backbone_net_ckpt_path

    exp_list = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '15', '17', '18', '19',
                '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '31']
    video_parent_path = 'data'
    video_output_parent_path = 'out'

    action_list = ['平举']
    exp_type_list = ["_lstm", "_hand_coco"]

    if mode == 'cal':
        all_video_list = get_video_list.get_all_video_list(video_parent_path, action_list)
        run_lstm.main(all_video_list, video_output_parent_path, checkpoint_path, stage_num, hm_channels, paf_channels,
                      use_bn, train_vgg, backbone_net_ckpt_path, val_anno_path, val_img_path, save_json_path, pd_video)
        get_pic_from_video.main(all_video_list, video_output_parent_path)
        run_hand.main(all_video_list, video_output_parent_path, hand_use_bn, hand_train_vgg, hand_checkpoint_path,
                      hand_backbone_net_ckpt_path)
        dis_anno.main_exp(all_video_list, video_parent_path, video_output_parent_path, exp_list, exp_type_list)
    elif mode == "test":
        right_count = 0
        threeD_list = []
        pre_list = []
        doc_accuracy = []
        doc_f1_score = []
        doc_precision = []
        doc_recall_score = []
        gt_list = threeD_list
        raw_ampli_list = []
        raw_ampli_gt_list = []
        raw_excel = pd.read_excel('doctor_rating_results_31.xlsx')
        doctor_data = raw_excel[1:28]
        doctor_data_array = doctor_data.iloc[0:27, 2:12].values
        doctor_data_array = np.transpose(doctor_data_array, axes=[1, 0])
        doctor_data_list = np.array(doctor_data_array, dtype=np.int).tolist()

        for i in exp_list:
            gt, pre, raw_amplitude, raw_amplitude_gt = smoother(
                'out/' + i + '/平举_hand_coco.json')
            raw_ampli_list.append(raw_amplitude)
            raw_ampli_gt_list.append(raw_amplitude_gt)
            threeD_list.append(gt)
            pre_list.append(pre)

        for doc_pre in doctor_data_list:
            doc_accuracy.append(
                metrics.accuracy_score(gt_list, doc_pre))
            doc_f1_score.append(
                metrics.f1_score(gt_list, doc_pre, average='macro'))
            doc_precision.append(
                metrics.precision_score(gt_list, doc_pre,
                                        average='macro'))
            doc_recall_score.append(
                metrics.recall_score(gt_list, doc_pre, average='macro'))
            metrics.confusion_matrix(gt_list, doc_pre)
        print(gt_list)
        print(pre_list)
        mean_squared_error = metrics.mean_squared_error(gt_list,
                                                        pre_list)
        accuracy = metrics.accuracy_score(gt_list, pre_list)
        f1_score = metrics.f1_score(gt_list, pre_list, average='macro')
        precision = metrics.precision_score(gt_list, pre_list,
                                            average='macro')
        recall_score = metrics.recall_score(gt_list, pre_list,
                                            average='macro')

        doctor_right_matrix = [[], [], [], [], []]
        chief_right_matrix = [[], [], [], [], []]
        pre_right_matrix = [[], [], [], [], []]
        doc_distribution = []
        chief_distribution = []
        pre_distribution = []
        for i in range(len(exp_list)):
            for doc_pre in doctor_data_list:
                if doc_pre[i] == gt_list[i]:
                    doctor_right_matrix[gt_list[i]].append(int(1))
                else:
                    doctor_right_matrix[gt_list[i]].append(int(0))
            if doctor_data_list[0][i] == gt_list[i]:
                chief_right_matrix[gt_list[i]].append(int(1))
            else:
                chief_right_matrix[gt_list[i]].append(int(0))
            if pre_list[i] == gt_list[i]:
                pre_right_matrix[gt_list[i]].append(int(1))
            else:
                pre_right_matrix[gt_list[i]].append(int(0))
        for level in chief_right_matrix:
            chief_distribution.append(sum(level) / len(level))
        for level in doctor_right_matrix:
            doc_distribution.append(sum(level) / len(level))
        for level in pre_right_matrix:
            pre_distribution.append(sum(level) / len(level))
        print('mean_squared_error: ' + str(mean_squared_error))
        print('method_accuracy: ' + str(accuracy))
        print('doctor_accuracy: ' + str(np.mean(doc_accuracy)))
        print('method_f1_score: ' + str(f1_score))
        print('doctor_f1_score: ' + str(np.mean(doc_f1_score)))
        print('method_precision: ' + str(precision))
        print('doctor_precision: ' + str(np.mean(doc_precision)))
        print('method_recall_score: ' + str(recall_score))
        print('doctor_recall_score: ' + str(np.mean(doc_recall_score)))
        print(doc_accuracy)
        print('doc_distribution: ' + str(doc_distribution))
        print('chief_distribution: ' + str(chief_distribution))
        print('pre_distribution: ' + str(pre_distribution))
    else:
        print("unknown mode")
