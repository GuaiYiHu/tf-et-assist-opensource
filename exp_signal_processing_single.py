import argparse
from exp_signal_processing import smoother
import dis_anno
import get_pic_from_video
import run_hand
import run_lstm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test codes for Openpose using Tensorflow')
    parser.add_argument('--mode', '-m', help='Test mode, "cal" or test', default="cal")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/fine_tone_doctor/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--video', required=True, type=str, default=None)
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

    video_output_parent_path = 'out'

    exp_type_list = ["_lstm", "_hand_coco"]
    video_list = []
    action_list = ['平举']
    video = args.video
    video_list.append(video)
    all_video_list = {}
    all_video_list.update({action_list[0]: video_list})

    run_lstm.main(all_video_list, video_output_parent_path, checkpoint_path, stage_num, hm_channels, paf_channels,
                  use_bn, train_vgg, backbone_net_ckpt_path, val_anno_path, val_img_path, save_json_path, pd_video)
    get_pic_from_video.main(all_video_list, video_output_parent_path)
    run_hand.main(all_video_list, video_output_parent_path, hand_use_bn, hand_train_vgg, hand_checkpoint_path,
                  hand_backbone_net_ckpt_path)
    dis_anno.main(all_video_list, video_output_parent_path, exp_type_list)
    dir_name = video.split('/')
    name = dir_name[-2]
    gt, pre, raw_amplitude, raw_amplitude_gt = smoother('out/' + name + '/平举_hand_coco.json')
    print("gt: ", gt)
    print("pre: ", pre)
    print("raw_amplitude: ", raw_amplitude)
    print("raw_amplitude_gt:", raw_amplitude_gt)
