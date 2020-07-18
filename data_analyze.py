import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import xlsxwriter
from sklearn.metrics import confusion_matrix

import get_video_list

name_list = []
time_list = []


# Get file status
def is_file_exist(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


# Get result from ground truth
def get_data_from_ground_truth(table_path):
    data_list = {}
    if is_file_exist(table_path):
        data = xlrd.open_workbook(table_path)
        sheet = data.sheets()[0]
        num_list = sheet.col_values(0)[2:]
        name_list = sheet.col_values(1)[2:]
        time_list = sheet.row_values(1)[2: 6]
        pose_list = list(filter(None, sheet.row_values(0)))

    for name_index, name in enumerate(name_list):
        pose_data_list = {}

        for pose_index, pose in enumerate(pose_list):

            time_data_list = {}

            for time_index, time in enumerate(time_list):
                amp = sheet.row_values(name_index + 2)[4 * pose_index + time_index + 2]

                time_data = {time: amp}
                time_data_list.update(time_data)

            pose_data = {pose: time_data_list}
            pose_data_list.update(pose_data)

        result = {name: {'num': num_list[name_index], 'result': pose_data_list}}
        data_list.update(result)

    return data_list


# Read all result from json
def get_all_result_from_video(video_parent_path, action_list, action_info, result_type_list):
    all_result_list = {}
    for result_type in result_type_list:
        result_list = {}
        all_video_list = get_video_list.get_all_video_list(video_parent_path, action_list)

        for video in all_video_list[action_list[0]]:
            dir_name = video.split('/')

            if dir_name[-2] not in name_list:
                name_list.append(dir_name[-2])

            if dir_name[-3] not in time_list:
                time_list.append(dir_name[-3])

            for name in name_list:
                pose_result_list = {}

                for action in action_list:
                    pose_list = action_info[action]['pose']
                    for pose in pose_list:
                        time_result_list = {}
                        for time in time_list:
                            json_path = video_parent_path + '/out/' + time + '/' + name + '/result/result_' + action + result_type + '.json'
                            if is_file_exist(json_path):
                                with open(json_path, 'r') as f:
                                    json_txt = json.load(f)
                                    amp = json_txt[pose]
                            else:
                                amp = ''

                            time_result = {time: amp}
                            time_result_list.update(time_result)

                        pose_data = {pose: time_result_list}
                        pose_result_list.update(pose_data)

                tmp_result = {name: {'result': pose_result_list}}
                result_list.update(tmp_result)
            result = {result_type: result_list}
            all_result_list.update(result)

    return all_result_list


def write_sheet(data_list, all_result_list, video_parent_path, result_type_list, pearson_pose_black_list):
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(video_parent_path + '/output.xlsx')
    for result_type in result_type_list:
        result_list = all_result_list[result_type]
        worksheet = workbook.add_worksheet('sheet' + result_type)

        # Write num and name title
        worksheet.write(1, 0, 'num')
        worksheet.write(1, 1, 'name')
        name_list = data_list.keys()
        for name_index, name in enumerate(name_list):
            num = data_list[name]['num']
            worksheet.write(name_index + 2, 0, num)
            worksheet.write(name_index + 2, 1, name)

        # Write pose and time title
        merge_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
        })

        pose_list = data_list[list(data_list.keys())[0]]['result'].keys()
        time_list = data_list[list(data_list.keys())[0]]['result'][list(pose_list)[0]].keys()

        for pose_index, pose in enumerate(pose_list):
            worksheet.merge_range(0, 4 * pose_index + 2, 0, 4 * pose_index + 5, pose, merge_format)
            for time_index, time in enumerate(time_list):
                worksheet.write(1, 4 * pose_index + time_index + 2, time)

        # Write level result
        for name_index, name in enumerate(name_list):
            for pose_index, pose in enumerate(pose_list):
                for time_index, time in enumerate(time_list):
                    if result_list.get(name) is not None:
                        if result_list.get(name).get('result').get(pose) is not None:
                            if result_list.get(name).get('result').get(pose).get(time) is not None:
                                result_value = result_list.get(name).get('result').get(pose).get(time)
                                worksheet.write(name_index + 2, 4 * pose_index + time_index + 2,
                                                getPoint(result_value))

        # Write accuracy result
        worksheet.merge_range(len(name_list) + 2, 0, len(name_list) + 2, 1, 'RMSE', merge_format)
        worksheet.merge_range(len(name_list) + 3, 0, len(name_list) + 3, 1, 'Pearson', merge_format)
        worksheet.merge_range(len(name_list) + 4, 0, len(name_list) + 4, 1, 'Spearman', merge_format)
        worksheet.merge_range(len(name_list) + 5, 0, len(name_list) + 5, 1, 'Kendall', merge_format)
        for pose_index, pose in enumerate(pose_list):
            x_list = []
            y_list = []
            for name in name_list:
                for time in time_list:
                    if data_list.get(name) is not None and result_list.get(name) is not None:
                        if data_list.get(name).get('result').get(pose) is not None and result_list.get(name).get(
                                'result').get(pose) is not None:
                            if is_data_valid(data_list.get(name).get('result').get(pose).get(time)) and is_data_valid(
                                    result_list.get(name).get('result').get(pose).get(time)):
                                x_point = int(data_list.get(name).get('result').get(pose).get(time))
                                y_point = getPoint(result_list.get(name).get('result').get(pose).get(time))
                                x_list.append(x_point)
                                y_list.append(y_point)

            if len(x_list) != 0 and len(y_list) != 0:
                # RMSE
                rmse = np.sqrt(sum((np.array(x_list) - np.array(y_list)) ** 2) / len(x_list))
                worksheet.merge_range(len(name_list) + 2, 4 * pose_index + 2, len(name_list) + 2,
                                      4 * pose_index + 5, round(rmse, 2), merge_format)

                X1 = pd.Series(x_list)
                Y1 = pd.Series(y_list)

                # Pearson
                if pose not in pearson_pose_black_list:
                    worksheet.merge_range(len(name_list) + 3, 4 * pose_index + 2, len(name_list) + 3,
                                          4 * pose_index + 5, round(X1.corr(Y1, method="pearson"), 2), merge_format)
                    worksheet.merge_range(len(name_list) + 4, 4 * pose_index + 2, len(name_list) + 4,
                                          4 * pose_index + 5, round(X1.corr(Y1, method="spearman"), 2), merge_format)
                    worksheet.merge_range(len(name_list) + 5, 4 * pose_index + 2, len(name_list) + 5,
                                          4 * pose_index + 5, round(X1.corr(Y1, method="kendall"), 2), merge_format)

    workbook.close()


def copy_video(data_list, all_result_list, video_parent_path, action_list, action_info, result_type_list):
    name_list = data_list.keys()
    result_list = all_result_list[result_type_list[-1]]
    pose_list = data_list[list(data_list.keys())[0]]['result'].keys()
    date_list = data_list[list(data_list.keys())[0]]['result'][list(pose_list)[0]].keys()

    for action in action_list:
        for date in date_list:
            for name in name_list:

                tmp_pose_list = action_info[action]['pose']

                err_flag = []
                for pose_index, pose in enumerate(tmp_pose_list):
                    if data_list.get(name) is not None and result_list.get(name) is not None:
                        if data_list.get(name).get('result').get(pose) is not None \
                                and result_list.get(name).get('result').get(pose) is not None:
                            if is_data_valid(data_list.get(name).get('result').get(pose).get(date)) \
                                    and is_data_valid(
                                result_list.get(name).get('result').get(pose).get(date)):
                                x_point = int(data_list.get(name).get('result').get(pose).get(date))
                                y_point = getPoint(result_list.get(name).get('result').get(pose).get(date))
                                err_flag.append(abs(x_point - y_point) <= 1)

                if False not in err_flag and len(err_flag) != 0:
                    copy_video_path = video_parent_path + '/video/' + action + '/data/' + date + '/' + name

                    if not os.path.exists(copy_video_path):
                        os.makedirs(copy_video_path)

                    file_path = video_parent_path + '/data/' + date + '/' + name + '/' + action + '.mp4'
                    if os.path.exists(file_path):
                        shutil.copy(file_path, copy_video_path)


# Get data's validity
def is_data_valid(value):
    return value is not None and value != ''


def draw_confusion_matrix(data_list, all_result_list, video_parent_path, result_type_list, pearson_pose_black_list):
    for result_type in result_type_list:
        result_list = all_result_list[result_type]
        pose_list = data_list[list(data_list.keys())[0]]['result'].keys()

        x_list = []
        y_list = []
        for pose_index, pose in enumerate(pose_list):
            if pose not in pearson_pose_black_list:
                for name in name_list:
                    for time in time_list:
                        if data_list.get(name) is not None and result_list.get(name) is not None:
                            if data_list.get(name).get('result').get(pose) is not None and result_list.get(name).get(
                                    'result').get(pose) is not None:
                                if is_data_valid(
                                        data_list.get(name).get('result').get(pose).get(time)) and is_data_valid(
                                        result_list.get(name).get('result').get(pose).get(time)):
                                    x_point = int(data_list.get(name).get('result').get(pose).get(time))
                                    y_point = getPoint(result_list.get(name).get('result').get(pose).get(time))
                                    x_list.append(x_point)
                                    y_list.append(y_point)

                if pose_index % 2 != 0 and pose_index != 0:
                    if len(x_list) != 0 and len(y_list) != 0:
                        f, ax = plt.subplots()
                        C2_origin = confusion_matrix(x_list, y_list, labels=[0, 1, 2, 3, 4])
                        C2 = np.array(C2_origin).astype(np.float).tolist()
                        for first_index in range(len(C2_origin)):
                            for second_index in range(len(C2_origin[first_index])):
                                C2[first_index][second_index] = C2_origin[first_index][second_index] / \
                                                                np.sum(C2_origin, axis=1)[first_index]

                        ax.set_title('confusion matrix')
                        ax.set_xlabel('prediction')
                        ax.set_ylabel('ground truth')
                        for first_index in range(len(C2)):
                            for second_index in range(len(C2[first_index])):
                                plt.text(first_index, second_index,
                                         "%0.2f" % (C2[second_index][first_index] * 100,) + '%', fontsize=12,
                                         va='center', ha='center')
                        plt.imshow(C2, cmap=plt.cm.Blues)
                        pose = pose.split('_')[-1]
                        plt.savefig(video_parent_path + '/matrix' + result_type + '_' + pose + '.jpg')
                    x_list = []
                    y_list = []


# Mark level
def getPoint(result):
    if result != '':
        result = float(result)
        if result < 0.5:
            return 0
        elif 0.5 <= result < 1:
            return 1
        elif 1 <= result < 2:
            return 2
        elif 2 <= result < 4:
            return 3
        else:
            return 4
    else:
        return ''


if __name__ == '__main__':
    video_parent_path = '/media/yzy/diskF/parkinson_data/data_cut_new'
    action_list = ['全身', '平举']
    action_info = {"全身": {'begin': 30, 'Wn': [0.2, 0.6], 'invalid_num': 45, 'pose': ['r_rest', 'l_rest']},
                   "平举": {'begin': 100, 'Wn': [0.2, 0.6], 'invalid_num': 45, 'pose': ['r_posture', 'l_posture']}}
    result_type_list = ['', '_hand_coco']
    pearson_pose_black_list = ['r_rest', 'l_rest']
    data_list = get_data_from_ground_truth(video_parent_path + '/mark_ground_truth.xlsx')
    all_result_list = get_all_result_from_video(video_parent_path, action_list, action_info, result_type_list)
    write_sheet(data_list, all_result_list, video_parent_path, result_type_list, pearson_pose_black_list)
    # copy_video(data_list, all_result_list, video_parent_path, action_list, action_info, result_type_list)
    draw_confusion_matrix(data_list, all_result_list, video_parent_path, result_type_list, pearson_pose_black_list)
