import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import vgg
from cut_body_parts import cut_body_part
from lstm_cpm import PafNet
from tensblur.smoother import Smoother

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_hand(all_video_list, video_output_parent_path, use_bn, train_vgg, checkpoint_path, backbone_net_ckpt_path):
    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_outputs, hm_channel_num=2, use_bn=use_bn)
    hm_pre, added_layers_out = net.gen_hand_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    # cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
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
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        logger.info('initialization done')
        action_list = all_video_list.keys()
        for action in action_list:
            video_list = all_video_list[action]
            for video in video_list:
                dir_name = video.split('/')
                name = dir_name[-2]
                save_path = video_output_parent_path + '/' + name

                anno_loader = cut_body_part(anno_file=save_path + '/' + action + '_lstm.json',
                                            coco_images=save_path + '/pics/')
                img_info = []
                anno_info = []
                for img, hand_list, img_meta, anno in tqdm(anno_loader.crop_part()):
                    for hand in hand_list:
                        position = hand['position']
                        ori_h = position[3] - position[1] + 1
                        ori_w = position[2] - position[0] + 1
                        peaks_origin, heatmap_origin = sess.run([tensor_peaks, hm_up, ],
                                                                feed_dict={raw_img: hand['hand'][np.newaxis, :, :, :],
                                                                           img_size: [ori_h, ori_w]})
                        re_origin = np.where(peaks_origin[0, :, :, 0] == np.max(peaks_origin[0, :, :, 0]))
                        peaks_flip, heatmap_flip = sess.run([tensor_peaks, hm_up, ], feed_dict={
                            raw_img: np.fliplr(hand['hand'][np.newaxis, :, :, :]),
                            img_size: [ori_h, ori_w]})
                        peaks_flip = np.fliplr(peaks_flip)
                        re_flip = np.where(peaks_flip[0, :, :, 0] == np.max(peaks_flip[0, :, :, 0]))
                        anno['keypoints'][hand['idx'] * 3] = int(position[0] + (re_origin[1][0] + re_flip[1][0]) / 2)
                        anno['keypoints'][hand['idx'] * 3 + 1] = int(
                            position[1] + (re_origin[0][0] + re_flip[0][0]) / 2)
                    anno_info.append(anno)
                    img_info.append(img_meta)
                ref = {"images": img_info, "annotations": anno_info}
                with open(save_path + '/' + action + '.json'.split('.')[0] + '_hand_coco' + '.json', "w") as f:
                    json.dump(ref, f)
                    print(
                        'writed to ' + save_path + '/' + action + '.json'.split('.')[0] + '_hand_coco' + '.json')


def main(all_video_list, video_output_parent_path, use_bn, train_vgg, checkpoint_path, backbone_net_ckpt_path):
    run_hand(all_video_list, video_output_parent_path, use_bn, train_vgg, checkpoint_path, backbone_net_ckpt_path)
