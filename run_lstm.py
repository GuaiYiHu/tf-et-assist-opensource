import copy
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import vgg
from common import CrossPartCocoPoseTrack
from estimator import PoseEstimator, TfPoseEstimator
from lstm_cpm import PafNet
from pose_dataset_pose_track import ParkinsonValDataLoader
from tensblur.smoother import Smoother

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def convert_anno(image_path, keypoints, img_size):
    annotations = []
    file_name = image_path.split('.')[0].split('/')[-1] + '_' + image_path.split('_')[-1]
    image_id = int(file_name.split('_')[-1])
    image = {"file_name": file_name, "id": image_id, "height": img_size[0], "width": img_size[1]}
    for i, keypoint in enumerate(keypoints):
        annotations.append(
            {"num_keypoints": 17, "keypoints": keypoint, "image_id": image_id, "id": int(image_id * 10 + i)})
    return image, annotations


def get_last_keypoints_from_bodies(bodys, keypoints_list):
    if len(bodys) == 0:
        keypoint = [0 for _ in range(51)]
    else:
        for human in bodys:
            keypoint = []
            for coco_part in keypoints_list:
                if coco_part is not None:
                    if coco_part.value not in human.body_parts.keys():
                        keypoint = keypoint + [0, 0, 0]
                        continue
                    keypoint = keypoint + [human.body_parts[coco_part.value].x,
                                           human.body_parts[coco_part.value].y, 1]
                else:
                    keypoint = keypoint + [0, 0, 0]
                    continue
    return keypoint


def run_lstm(all_video_list, video_output_parent_path, checkpoint_path, stage_num, hm_channels, paf_channels, use_bn,
             train_vgg,
             backbone_net_ckpt_path, val_anno_path, val_img_path, save_json_path, pd_video):
    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    vgg_out = []
    with slim.arg_scope(vgg.vgg_arg_scope()):
        for i in range(stage_num - 1):
            vgg_outputs, end_points = vgg.vgg_19(img_normalized[:, i, :, :, :])
            vgg_out.append(vgg_outputs)

    # get net graph
    logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_out, stage_num=stage_num, hm_channel_num=hm_channels,
                 paf_channel_num=paf_channels, use_bn=use_bn)
    # net = PafNet(inputs_x=vgg_out, use_bn=use_bn)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                            tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if train_vgg:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)
    logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    action_list = all_video_list.keys()
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...' + tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        # saver.restore(sess, checkpoint_path + 'model-65000')
        logger.info('initialization done')

        if val_anno_path is None:
            keypoints_list = [CrossPartCocoPoseTrack.Nose, None, None, None, None,
                              CrossPartCocoPoseTrack.LShoulder, CrossPartCocoPoseTrack.RShoulder,
                              CrossPartCocoPoseTrack.LElbow, CrossPartCocoPoseTrack.RElbow,
                              CrossPartCocoPoseTrack.LWrist, CrossPartCocoPoseTrack.RWrist,
                              CrossPartCocoPoseTrack.LHip, CrossPartCocoPoseTrack.RHip,
                              CrossPartCocoPoseTrack.LKnee, CrossPartCocoPoseTrack.RKnee,
                              CrossPartCocoPoseTrack.LAnkle, CrossPartCocoPoseTrack.RAnkle]

            for action in action_list:
                video_list = all_video_list[action]
                for video in video_list:
                    dir_name = video.split('/')
                    name = dir_name[-2]

                    save_result_path = video_output_parent_path + '/' + name
                    isExists = os.path.exists(save_result_path)

                    if not isExists:
                        os.makedirs(save_result_path)

                    coco_images = []
                    coco_annos = []
                    if video is not None:
                        cap = cv2.VideoCapture(video)
                    else:
                        cap = cv2.VideoCapture(0)
                        # cap = cv2.VideoCapture('http://admin:admin@192.168.1.52:8081')
                    _, image = cap.read()
                    if image is None:
                        logger.error("Can't read video")
                        sys.exit(-1)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 7
                    w = 500
                    h = int(w * (ori_h / ori_w))
                    size = [h, w]

                    if save_result_path is not None:
                        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        video_saver = cv2.VideoWriter(save_result_path + '/' + action + '.MOV', fourcc, fps,
                                                      (ori_w, ori_h))
                        logger.info('record vide to %s' % save_result_path + '/' + action + '.MOV')
                    logger.info('fps@%f' % fps)
                    time_n = time.time()

                    # _, image = cap.read()
                    # image = np.transpose(image, [1, 0, 2])
                    img = np.array(cv2.resize(image, (w, h)))
                    img = img[np.newaxis, np.newaxis, :]
                    img = np.repeat(img, 5, axis=1)
                    img_count = 0
                    for _ in tqdm(range(int(frame_count - 9))):
                        img = np.delete(img, 0, axis=1)
                        try:
                            _, image = cap.read()
                            # image = np.transpose(image, [1, 0, 2])
                        except StopIteration:
                            print('Done!')
                        img_to_save = image
                        image = np.array(cv2.resize(image, (w, h)))
                        # cv2.imshow('raw', image)
                        # img_corner = np.array(cv2.resize(image, (360, int(360*(ori_h/ori_w)))))
                        img = np.concatenate((img, image[np.newaxis, np.newaxis, :]), axis=1)
                        load_img = img  # [:, ::2, :, :, :]
                        peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                             feed_dict={raw_img: load_img, img_size: size})
                        # plt.imshow(heatmap[0, :, :, 15])
                        # plt.show()
                        bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                        image = TfPoseEstimator.draw_humans(img_to_save, bodys, imgcopy=False)
                        fps = round(1 / (time.time() - time_n), 2)
                        image = cv2.putText(image, str(fps) + 'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                            (255, 255, 255))
                        video_saver.write(image)

                        keypoints = []
                        if len(bodys) == 0:
                            keypoint = [0 for _ in range(51)]
                        else:
                            for human in bodys:
                                keypoint = []
                                for coco_part in keypoints_list:
                                    if coco_part is not None:
                                        if coco_part.value not in human.body_parts.keys():
                                            keypoint = keypoint + [0, 0, 0]
                                            continue
                                        keypoint = keypoint + [int(human.body_parts[coco_part.value].x * ori_w + 0.5),
                                                               int(human.body_parts[coco_part.value].y * ori_h + 0.5),
                                                               1]
                                    else:
                                        keypoint = keypoint + [0, 0, 0]
                                        continue
                                # im_show = cv2.circle(image, )
                        keypoints.append(keypoint)
                        image_name = video + '_' + str(img_count)
                        coco_image, coco_anno = convert_anno(image_name, keypoints, [ori_h, ori_w])
                        coco_images.append(coco_image)
                        coco_annos += coco_anno
                        img_count += 1
                    if save_result_path is not None:
                        ref = {"images": coco_images, "annotations": coco_annos}
                        with open(save_result_path + "/" + action + ".json", "w") as f:
                            json.dump(ref, f)
                            print('writed to ' + save_result_path + "/" + action + ".json")

        else:
            dataloader = ParkinsonValDataLoader(path=val_anno_path, img_path=val_img_path,
                                                result_save_path=save_json_path, image_num=9)
            time_n = time.time()
            for load_img, image_id in tqdm(dataloader.get_data()):
                size = (load_img.shape[2], load_img.shape[3])
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: load_img[:, ::2, :, :, :], img_size: size})
                # plt.imshow(heatmap[0, :, :, 15])
                # plt.show()
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                out_img = copy.deepcopy(load_img[0, -1, :, :, :])
                image = TfPoseEstimator.draw_humans(out_img, bodys, imgcopy=False)
                fps = round(1 / (time.time() - time_n), 2)
                image = cv2.putText(image, str(fps) + 'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 255, 255))
                time_n = time.time()
                if pd_video is not None:
                    pass
                    # image[27:img_corner.shape[0]+27, :img_corner.shape[1]] = img_corner  # [3:-10, :]
                # cv2.imshow(' ', image)
                # cv2.waitKey(1)
                # keypoints = []
                # keypoints.append(get_last_keypoints_from_bodies(bodys, keypoints_list))
                # img_info, anno_info = get_coco_val_anno(keypoints=keypoints, image_id=image_id)
                dataloader.add_result(bodys=bodys, image_id=image_id)
            average_dis_error, average_vec_error, dis_var, vec_var = dataloader.caculate_accuracy()
            print('average_dis_error: ' + str(average_dis_error) + 'dis_var: ' + str(dis_var))
            print('average_vec_error: ' + str(average_vec_error) + 'vec_var: ' + str(vec_var))
    tf.reset_default_graph()


def main(all_video_list, video_output_parent_path, checkpoint_path, stage_num, hm_channels, paf_channels, use_bn,
         train_vgg, backbone_net_ckpt_path, val_anno_path, val_img_path, save_json_path, pd_video):
    run_lstm(all_video_list, video_output_parent_path, checkpoint_path, stage_num, hm_channels, paf_channels, use_bn,
             train_vgg,
             backbone_net_ckpt_path, val_anno_path, val_img_path, save_json_path, pd_video)
