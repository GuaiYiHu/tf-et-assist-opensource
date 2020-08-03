import copy
import json
import os

import numpy as np
import pywt
import scipy as sc
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from scipy import signal

import get_video_list

font = FontProperties(fname=r"simsun.ttc", size=10)


def caculate_frequency(points, fps=30, weight=1):
    cwtmatr1, freqs1 = pywt.cwt(points, np.arange(1, 100), 'cgau8', 1 / fps)
    # cwtmatr1= pywt.dwt(sf[:i + fps, keypoint_index], 'db2')
    fps = int(fps * weight)

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


def caculate_max_amplitude(amplitude, frequency_list, maintain_time_threshold=1.17, fps=30):
    maintain_frame = int(maintain_time_threshold * fps)
    windows_num = amplitude.shape[0] - maintain_frame
    slid_windows = np.zeros(shape=[windows_num, maintain_frame])
    for i in range(windows_num):
        slid_windows[i, :] = amplitude[i:i + maintain_frame]
    min_in_window = np.nanmin(slid_windows, axis=1)
    for i, a in enumerate(list(min_in_window)):
        if frequency_list[i] <= 3 or frequency_list[i] == float('inf'):
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
    return a_inter, peaks_inter, valleys_inter


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


def smoother(video_output_parent_path, all_video_list, action_info, result_type_list):
    action_list = all_video_list.keys()
    for action in action_list:
        video_list = all_video_list[action]
        for video in video_list:
            print(video)
            dir_name = video.split('/')
            name = dir_name[-2]

            save_path = video_output_parent_path + '/' + name

            for result_type in result_type_list:
                path = save_path + "/" + action + '.json'.split('.')[0] + result_type + '.json'
                data = json.load(open(path, 'r'))
                draw_human_data = copy.deepcopy(data)
                fps = 30
                begin = action_info[action]['begin']
                finish = 300
                keypoint_index = 0
                points = []
                for frame in data['annotations']:
                    points.append(frame['keypoints'])

                if len(points) >= action_info[action]['invalid_num']:
                    l_wrist_x = np.array(points)[begin:, 27]
                    l_wrist_y = np.array(points)[begin:, 28]
                    r_wrist_x = np.array(points)[begin:, 30]
                    r_wrist_y = np.array(points)[begin:, 31]
                    l_wrist = np.array(points)[begin:, 27:29]
                    r_wrist = np.array(points)[begin:, 30:32]
                    # human_points_x = np.array(points)[:300, 28]
                    l_shoulder = np.array(points)[begin:, 15:17]
                    r_shoulder = np.array(points)[begin:, 18:20]
                    l_elbow = np.array(points)[begin:, 21:23]
                    r_elbow = np.array(points)[begin:, 24:26]
                    l_r_shoulder_distance = np.mean(np.sqrt(np.sum(np.square(l_shoulder - r_shoulder), axis=1)))
                    l_shoulder_elbow_distance = np.mean(np.sqrt(np.sum(np.square(l_shoulder - l_elbow), axis=1)))
                    r_shoulder_elbow_distance = np.mean(np.sqrt(np.sum(np.square(r_shoulder - r_elbow), axis=1)))
                    l_elbow_wrist_distance = np.mean(np.sqrt(np.sum(np.square(l_elbow - l_wrist), axis=1)))
                    r_elbow_wrist_distance = np.mean(np.sqrt(np.sum(np.square(r_elbow - r_wrist), axis=1)))
                    l_arm = l_shoulder_elbow_distance + l_elbow_wrist_distance
                    r_arm = r_shoulder_elbow_distance + r_elbow_wrist_distance
                    # for idx, num in enumerate(human_points):
                    #     if num == 0:
                    #         n=idx
                    #         while human_points[n] == 0:
                    #             n += 1
                    #         if idx == 0:
                    #             human_points[idx] = human_points[n]
                    #         else:
                    #             human_points[idx] = int((human_points[idx - 1] + human_points[n]) / 2 + 0.5)
                    l_wrist_x = inter_miss_point(l_wrist_x)
                    l_wrist_y = inter_miss_point(l_wrist_y)
                    r_wrist_x = inter_miss_point(r_wrist_x)
                    r_wrist_y = inter_miss_point(r_wrist_y)

                    # x = np.linspace(0, 2, 120)
                    # f_gt = []
                    # a_gt = []
                    #
                    # points = 0 * np.sin(2 * np.pi * 3 * x[:100])
                    # f_gt.extend([3] * 100)
                    # a_gt.extend([1] * 100)
                    #
                    # points = np.concatenate((points, 2 * np.sin(2 * np.pi * 10 * np.linspace(0, 0.3, 18))), axis=0)
                    # f_gt.extend([10] *18)
                    # a_gt.extend([1] * 18)
                    #
                    # points = np.concatenate((points, 0 * np.sin(2 * np.pi * 3 * x[:100])), axis=0)
                    # f_gt.extend([3] * 100)
                    # a_gt.extend([1] * 100)
                    #
                    # points = np.concatenate((points, 3 * np.sin(2 * np.pi * 6 * x)), axis=0)
                    # f_gt.extend([6] * 120)
                    # a_gt.extend([1] * 120)
                    #
                    # points = np.concatenate((points, 0 * np.sin(2 * np.pi * 6 * x)), axis=0)
                    # f_gt.extend([6] * 120)
                    # a_gt.extend([5] * 120)
                    #
                    # points = np.concatenate((points, 4 * np.sin(2 * np.pi * 3 * x)), axis=0)
                    # f_gt.extend([3] * 120)
                    # a_gt.extend([5] * 120)

                    # white_noise = np.random.normal(0, 0.75, len(points))

                    # noise_signal = points + white_noise
                    # points = points[:, np.newaxis]
                    # # points[:, 0::3] = np.true_divide(points[:, 0::3], data['images'][0]['width'])
                    # # points[:, 1::3] = np.true_divide(points[:, 1::3], data['images'][0]['height'])
                    b, a = signal.butter(1, action_info[action]['Wn'], 'bandpass')
                    # sf = signal.filtfilt(b, a, human_points, axis=0)
                    l_wrist_x = signal.filtfilt(b, a, l_wrist_x, axis=0)
                    l_wrist_y = signal.filtfilt(b, a, l_wrist_y, axis=0)
                    r_wrist_x = signal.filtfilt(b, a, r_wrist_x, axis=0)
                    r_wrist_y = signal.filtfilt(b, a, r_wrist_y, axis=0)
                    # l_wrist_x = np.append(np.flipud(l_wrist_x), l_wrist_x)
                    # l_wrist_y = np.append(np.flipud(l_wrist_y), l_wrist_y)
                    # r_wrist_x = np.append(np.flipud(r_wrist_x), r_wrist_x)
                    # r_wrist_y = np.append(np.flipud(r_wrist_y), r_wrist_y)

                    # M = 64  # 滤波器的阶数
                    #
                    # mu = 0.0001  # 步长因子
                    #
                    # xs = human_points
                    #
                    # xn = xs  # 原始输入端的信号为被噪声污染的正弦信号
                    #
                    # dn = points  # 对于自适应对消器，用dn作为期望
                    #
                    # # 调用LMS算法
                    # itr = len(points)
                    # (yn, en) = LMS(xn, dn, M, mu, itr)
                    # yn = yn * 20
                    # sf = points
                    # plt.figure(figsize=(4, 1))

                    # sf = sf[:, np.newaxis]
                    try:
                        l_wrist_x_amplitude, l_wrist_x_peaks, l_wrist_x_valleys = caculate_amplitude(l_wrist_x)
                        l_wrist_y_amplitude, l_wrist_y_peaks, l_wrist_y_valleys = caculate_amplitude(l_wrist_y)
                        r_wrist_x_amplitude, r_wrist_x_peaks, r_wrist_x_valleys = caculate_amplitude(r_wrist_x)
                        r_wrist_y_amplitude, r_wrist_y_peaks, r_wrist_y_valleys = caculate_amplitude(r_wrist_y)
                        l_wrist_amplitude = np.sqrt(
                            np.square(l_wrist_x_amplitude * 1.7) + np.square(l_wrist_y_amplitude))
                        r_wrist_amplitude = np.sqrt(
                            np.square(r_wrist_x_amplitude * 1.7) + np.square(r_wrist_y_amplitude))
                        l_wrist_y_frenquency = caculate_frequency(l_wrist_y)
                        r_wrist_y_frenquency = caculate_frequency(r_wrist_y)

                        l_wrist_maintain_amplitude = caculate_max_amplitude(l_wrist_amplitude,
                                                                            l_wrist_y_frenquency) / l_r_shoulder_distance * 40
                        r_wrist_maintain_amplitude = caculate_max_amplitude(r_wrist_amplitude,
                                                                            r_wrist_y_frenquency) / l_r_shoulder_distance * 40
                        # l_wrist_maintain_amplitude = caculate_max_amplitude(l_wrist_amplitude, l_wrist_y_frenquency) / l_arm * 50
                        # r_wrist_maintain_amplitude = caculate_max_amplitude(r_wrist_amplitude, r_wrist_y_frenquency) / r_arm * 50
                        print('average l_wrist_x_amplitude: ' + str(np.mean(l_wrist_x_amplitude)))
                        print('average l_wrist_y_amplitude: ' + str(np.mean(l_wrist_y_amplitude)))
                        print('average l_wrist_amplitude: ' + str(np.mean(l_wrist_amplitude)))
                        print('average r_wrist_x_amplitude: ' + str(np.mean(r_wrist_x_amplitude)))
                        print('average r_wrist_y_amplitude: ' + str(np.mean(r_wrist_y_amplitude)))
                        print('average r_wrist_amplitude: ' + str(np.mean(r_wrist_amplitude)))
                        print('l_wrist_maintain_amplitude: ' + str(l_wrist_maintain_amplitude))
                        print('r_wrist_maintain_amplitude: ' + str(r_wrist_maintain_amplitude))
                        ref = {action_info[action]['pose'][0]: str(r_wrist_maintain_amplitude),
                               action_info[action]['pose'][1]: str(l_wrist_maintain_amplitude)}
                    except:
                        ref = None
                else:
                    ref = {action_info[action]['pose'][0]: '', action_info[action]['pose'][1]: ''}

                if ref is not None:
                    with open(save_path + '/result_' + action + result_type + '.json', 'w') as f:
                        json.dump(ref, f)

            # return
            #
            # save_result_path = video_path + '/plt'
            # isExists = os.path.exists(save_result_path)
            #
            # if not isExists:
            #     os.makedirs(save_result_path)
            #
            # starts, stops = find_wave_start_stop(l_wrist_y_amplitude, f_number=5, threshold=2.5)
            # # maintain_amplitude = caculate_max_amplitude(l_wrist_amplitude) / l_r_shoulder_distance * 35
            # for i in tqdm.tqdm(range(len(l_wrist_y))):
            #     plt.figure(figsize=(7.2, 10.8))
            #     ap = plt.subplot(6, 1, 1)
            #     plt.plot(l_wrist_y[:i], 'blue')
            #     ap.set_title('左手原始信号', fontproperties=font, fontsize=20)
            #
            #     # bp = plt.subplot(8, 1, 2)
            #     # plt.plot(l_wrist_y[:i], 'k')
            #     # plt.plot(l_wrist_y_peaks[:i], 'k')
            #     # plt.plot(l_wrist_y_valleys[:i], 'k')
            #     # # plt.plot(starts, list(0 * np.array(starts)), "bo")
            #     # # plt.plot(stops, list(0 * np.array(stops)), "ro")
            #     # bp.set_title('左手包络面', fontproperties=font)
            #
            #     cp = plt.subplot(6, 1, 2)
            #     plt.plot(l_wrist_amplitude[:i] / l_r_shoulder_distance * 40, 'blue')
            #     cp.set_title('左手幅值：' + str(round(l_wrist_amplitude[i] / l_r_shoulder_distance * 40, 2)), fontproperties=font, fontsize=20)
            #
            #     dp = plt.subplot(6, 1, 3)
            #     plt.plot(l_wrist_y_frenquency[:i], 'blue')
            #     dp.set_title('左手频率：' + str(round(l_wrist_y_frenquency[i], 2)), fontproperties=font, fontsize=20)
            #
            #     ep = plt.subplot(6, 1, 4)
            #     plt.plot(r_wrist_y[:i], 'blue')
            #     ep.set_title('右手原始信号', fontproperties=font, fontsize=20)
            #
            #     # fp = plt.subplot(8, 1, 6)
            #     # plt.plot(r_wrist_y[:i], 'k')
            #     # plt.plot(r_wrist_y_peaks[:i], 'k')
            #     # plt.plot(r_wrist_y_valleys[:i], 'k')
            #     # # plt.plot(starts, list(0 * np.array(starts)), "bo")
            #     # # plt.plot(stops, list(0 * np.array(stops)), "ro")
            #     # fp.set_title('右手包络面', fontproperties=font)
            #
            #     gp = plt.subplot(6, 1, 5)
            #     plt.plot(r_wrist_amplitude[:i] / l_r_shoulder_distance * 40, 'blue')
            #     gp.set_title('右手幅值：' + str(round(r_wrist_amplitude[i] / l_r_shoulder_distance * 40, 2)), fontproperties=font, fontsize=20)
            #
            #     hp = plt.subplot(6, 1, 6)
            #     plt.plot(r_wrist_y_frenquency[:i], 'blue')
            #     hp.set_title('右手频率：' + str(round(r_wrist_y_frenquency[i], 2)), fontproperties=font, fontsize=20)
            #
            #     # bp = plt.subplot(4, 1, 2)
            #     # plt.plot(np.array(l_wrist_y))
            #     # bp.set_title('原始信号R', fontproperties=font)
            #     # cp = plt.subplot(4, 1, 3)
            #     # plt.plot(l_wrist_y)
            #     # plt.plot(l_wrist_y_peak   s)
            #     # plt.plot(l_wrist_y_valleys)
            #     # plt.plot(starts, list(0*np.array(starts)), "bo")
            #     # plt.plot(stops, list(0*np.array(stops)), "ro")
            #     # cp.set_title('L包络面', fontproperties=font)
            #     # dp = plt.subplot(4, 1, 4)
            #     # plt.plot(l_wrist_y)
            #     # plt.plot(l_wrist_y_peaks)
            #     # plt.plot(l_wrist_y_valleys)
            #     # plt.plot(starts, list(0 * np.array(starts)), "bo")
            #     # plt.plot(stops, list(0 * np.array(stops)), "ro")
            #     # cp.set_title('L包络面', fontproperties=font)
            #     # dp = plt.subplot(4, 1, 4)
            #     # plt.plot(r_wrist_amplitude / l_r_shoulder_distance * 35)
            #     # dp.set_title('幅值', fontproperties=font)
            #     plt.tight_layout()
            #     plt.savefig(video_path + '/plt/plt_body' + str(i) + '.png')
            #     plt.close()
            #
            # create_video(video_path)
            # # plt.show()
            # return
            # # peaks_inter = interp1d([i for i in range(len(peaks))], peaks, kind='linear')
            # # for i, keypoints in enumerate(sf):
            # #     data['annotations'][i]['keypoints'] = list(keypoints)
            # # ref = {"images": data['images'], "annotations": data['annotations']}
            # # save_path = path.split('.')[0]+'_smooth.json'
            # # with open(save_path, "w") as f:
            # #     json.dump(ref, f)
            # #     print('writed to ' + save_path)
            # a_list = []
            # frequency_list = [0] * (fps - 1)
            # a_list = [0] * (fps - 1)
            # cwt_frequency_list = []
            # r_frequency_list = [0] * (fps - 1)
            # # plt.matshow(np.abs(cwtmatr1))
            #
            # # plt.pcolormesh(t, freqs, coef)
            # # plt.contourf(np.arange(len(sf[:, keypoint_index])), freqs1, abs(cwtmatr1))
            # # plt.title('time-frequency relationship of signal_1')
            # # plt.xlabel('Time/second')
            # # plt.ylabel('Frequency/Hz')
            # # coeffs = pywt.wavedec(sf[:, keypoint_index], 'db4', level=6)  # 4阶小波分解
            # # # x = [i for i in range(len(sf[:, keypoint_index]))]
            # # ya4 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 0, 0]).tolist(), 'db4')
            # # yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0, 0, 0]).tolist(), 'db4')
            # # yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0, 0]).tolist(), 'db4')
            # # yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0, 0]).tolist(), 'db4')
            # # yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1, 0, 0]).tolist(), 'db4')
            # # yd5 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 1, 0]).tolist(), 'db4')
            # # yd6 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 0, 1]).tolist(), 'db4')
            # #
            # # plt.figure(figsize=(12, 12))
            # # plt.subplot(611)
            # # plt.plot(sf[:, keypoint_index])
            # # plt.title('original signal')
            # # plt.subplot(612)
            # # plt.plot(ya4)
            # # plt.title('approximated component in level 4')
            # # plt.subplot(613)
            # # plt.plot(yd4)
            # # plt.title('detailed component in level 4')
            # # plt.subplot(614)
            # # plt.plot(yd3)
            # # plt.title('detailed component in level 3')
            # # plt.subplot(615)
            # # plt.plot(yd2)
            # # plt.title('detailed component in level 2')
            # # plt.subplot(616)
            # # plt.plot(yd1)
            # # plt.title('detailed component in level 1')
            # # plt.tight_layout()
            # # plt.show()
            #
            #
            # plt.ion()
            # plt.figure(figsize=(19.2, 21.6), dpi=50)
            # plt.subplots_adjust(left=0.06, bottom=0.06, right=0.96, top=0.93,
            #                     wspace=0.13, hspace=0.33)
            # for i in range(l_wrist_y.shape[0]-fps):
            #     # img = draw_human(draw_human_data['images'][i+fps], draw_human_data['annotations'][i+fps])
            #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     plt.clf()
            #     xf = np.fft.rfft(l_wrist_y[i:i+fps]) / fps
            #     freqs = np.linspace(0, fps / 2, fps / 2 + 1)
            #     xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e1000))
            #     try:
            #         f = np.where(xfp == np.max(xfp))[0][0]
            #     except:
            #         f = 4
            #     a_fft = np.max(np.abs(xf))
            #     print(a)
            #     try:
            #         cwtmatr1, freqs1 = pywt.cwt(l_wrist_y[:i+fps], np.arange(1, 100), 'cgau8', 1 / fps)
            #         # cwtmatr1= pywt.dwt(sf[:i + fps, keypoint_index], 'db2')
            #
            #         cwt_frequency_list = [np.inf]
            #         cwt_a_list = []
            #         for j in range(len(cwtmatr1[0, :i+fps])):
            #             cwt_abs = np.abs(cwtmatr1[:, j])
            #             max_cwt_abs = np.max(cwt_abs)
            #             try:
            #                 cwt_f_index = np.where(cwt_abs == max_cwt_abs)[0][0]
            #                 cwt_f = freqs1[cwt_f_index]
            #             except:
            #                 cwt_f = 5
            #             cwt_frequency_list.append(cwt_f)
            #             cwt_a_list.append(max_cwt_abs)
            #     except:
            #         pass
            #     # print(np.max(xfp))
            #     tf.reset_default_graph()
            #     loss, a, w, u, shift, x = sin_regression(l_wrist_y[i:i+fps], w=np.array(2*np.pi*f), fps=fps)
            #     train = tf.train.AdamOptimizer(learning_rate=1e-1, epsilon=1e-8).minimize(loss)
            #
            #     config = tf.ConfigProto()
            #     config.gpu_options.allow_growth = True
            #     with tf.Session(config=config) as sess:
            #         sess.run(tf.group(tf.global_variables_initializer()))
            #         for j in range(100):
            #             # sess.run(train)
            #             current_loss, a_p, w_p, u_p, shift_pre, x_p, _ = sess.run([loss, a, w, u, shift, x, train])
            #         # print(a_p, w_p, u_p, shift_pre, current_loss)
            #         sess.close()
            #     # print(f[0])
            #     # print(a_p, w_p, u_p, shift_pre, current_loss)
            #     # sin = a_p * np.sin(2*np.pi*f[0][0]*np.linspace(0, 1, fps)+u_p) + shift_pre
            #     a_list.append(a_p)
            #     r_frequency_list.append(w_p / np.pi / 2)
            #     frequency_list.append(f)
            #     # cwt_frequency_list.append(cwt_f)
            #     # sin = a_p * np.sin(w_p*np.linspace(0, 1, fps)+u_p) + shift_pre
            #     # xf_all = np.fft.rfft(sf[0:i + fps, keypoint_index]) / sf[0:i + fps, keypoint_index].shape[0]
            #     # freqs_all = np.linspace(0, fps / 2, sf[0:i + fps, keypoint_index].shape[0] / 2 + 1)
            #     # xfp_all = 20 * np.log10(np.clip(np.abs(xf_all), 1e-20, 1e1000))
            #
            #     # xf_ori = np.fft.rfft(points[0:i + fps, keypoint_index]) / points[0:i + fps, keypoint_index].shape[0]
            #     # xfp_ori = 20 * np.log10(np.clip(np.abs(xf_ori), 1e-20, 1e1000))
            #     # freqs_ori = np.linspace(0, fps / 2, points[0:i + fps, keypoint_index].shape[0] / 2 + 1)
            #
            #     gs = GridSpec(12, 1)
            #
            #     egraphic = plt.subplot(gs[:2, 0])
            #     egraphic.set_title('原始信号', fontproperties=font)
            #     egraphic.plot(np.array(np.array(points)[begin:finish, 28][0:i + fps]))
            #
            #
            #     # ngraphic = plt.subplot(gs[2:4, 0])
            #     # ngraphic.set_title('噪声污染信号', fontproperties=font)
            #     # ngraphic.plot(np.array(noise_signal[0:i + fps]))
            #
            #
            #     # agraphic = plt.subplot(gs[8, 0])
            #     # agraphic.set_title('滑窗抖动曲线', fontproperties=font)
            #     # agraphic.plot(np.array(sf[i:i+fps:, keypoint_index]))
            #     # agraphic.plot'original', 'regression'], loc = 'upper right')
            #     #
            #     #
            #     #         bgraphic = plt.subplot(gs[2:4, 0])
            #     #         bgraphic.set_title('滤波信号', fontproperties=font)(sin)
            #     # agraphic.legend([        bgraphic.plot(np.array(r_wrist_y[0:i + fps:]))
            #     # bgraphic.plot(r_wrist_y_peaks[0:i + fps])
            #     # bgraphic.plot(r_wrist_y_valleys[0:i + fps])
            #     plt.xlim(left=0)
            #
            #     # cgraphic = plt.subplot(gs[8, 1])
            #     # cgraphic.set_title('滑窗抖动曲线频谱', fontproperties=font)
            #     # cgraphic.plot(freqs, xfp)
            #
            #
            #     # dgraphic = plt.subplot(gs[7, 1])
            #     # dgraphic.set_title('滤波信号频谱', fontproperties=font)
            #     # dgraphic.plot(freqs_all, xfp_all)
            #
            #
            #     # fgraphic = plt.subplot(gs[6, 1])
            #     # fgraphic.set_title('原始信号频谱', fontproperties=font)
            #     # fgraphic.plot(freqs_ori, xfp_ori)
            #
            #
            #     ggraphic = plt.subplot(gs[4:6, 0])
            #     ggraphic.set_title('振幅曲线', fontproperties=font)
            #     # ggraphic.plot(a_list)
            #     # ggraphic.plot(cwt_a_list)
            #     # ggraphic.plot(a_gt[:i+fps])
            #     ggraphic.plot(r_wrist_amplitude[:i+fps] / l_r_shoulder_distance * 40)
            #     # plt.ylim(0, 4)
            #
            #
            #     hgraphic = plt.subplot(gs[6:8, 0])
            #     hgraphic.set_title('频率曲线', fontproperties=font)
            #     hgraphic.plot(frequency_list)
            #     hgraphic.plot(cwt_frequency_list)
            #     hgraphic.plot(r_frequency_list)
            #     # hgraphic.plot(f_gt[:i+fps])
            #     # plt.ylim(0, 7)
            #
            #     # igraphic = plt.subplot(gs[:6, :])
            #     # igraphic.set_title('骨架', fontproperties=font)
            #     # igraphic.imshow(img)
            #
            #     # igraphic = plt.subplot(gs[10:12, :])
            #     # igraphic.set_title('时频图', fontproperties=font)
            #     # try:
            #     #     igraphic.contourf(np.arange(0, len(sf[65:i+fps, keypoint_index])), freqs1, abs(cwtmatr1))
            #     # except:
            #     #     pass
            #     # plt.ylim(0,  10)
            #     # plt.pause(0.01)
            #
            #     save_result_path = video_path + '/plt'
            #     isExists = os.path.exists(save_result_path)
            #
            #     if not isExists:
            #         os.makedirs(save_result_path)
            #
            #     plt.savefig(video_path + '/plt/plt_body' + str(i) + '.jpg')
            #
            # plt.ioff()  # 关闭画图的窗口，即关闭交互模式
            # # plt.clf()
            # # agraphic.plot(np.array(sf[:, 30]))
            # # xf = np.fft.rfft(sf[:, 30]) / sf.shape[0]
            # # freqs = np.linspace(0, 60 / 2, sf.shape[0] / 2 + 1)
            # # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e1000))
            # # bgraphic.plot(freqs, xfp)
            # # plt.show()
            #     # plt.show()
            #     # plt.plot(freqs, xfp)
            #     # plt.show()
            #
            #
            # # return save_path


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
            # cwtmatr1= pywt.dwt(sf[:i + fps, keypoint_index], 'db2')

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
            # cwtmatr1= pywt.dwt(sf[:i + fps, keypoint_index], 'db2')

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
    # plt.plot(left_hand_distances)
    # plt.plot(right_hand_distances)
    # plt.plot(cwt_frequency_list)
    plt.show()
