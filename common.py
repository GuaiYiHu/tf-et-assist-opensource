from enum import Enum

import cv2
import tensorflow as tf

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


class PostTrackPart(Enum):
    Nose = 0
    Upper_Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Head_top = 14
    Background = 15


class CrossPartCocoPoseTrack(Enum):
    Nose = 0
    Upper_Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Background = 14


class MPIIPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    RWrist = 6
    RElbow = 7
    RShoulder = 8
    LShoulder = 9
    LElbow = 10
    LWrist = 11
    Neck = 12
    Head = 13

    @staticmethod
    def from_coco(human):
        # t = {
        #     MPIIPart.RAnkle: CocoPart.RAnkle,
        #     MPIIPart.RKnee: CocoPart.RKnee,
        #     MPIIPart.RHip: CocoPart.RHip,
        #     MPIIPart.LHip: CocoPart.LHip,
        #     MPIIPart.LKnee: CocoPart.LKnee,
        #     MPIIPart.LAnkle: CocoPart.LAnkle,
        #     MPIIPart.RWrist: CocoPart.RWrist,
        #     MPIIPart.RElbow: CocoPart.RElbow,
        #     MPIIPart.RShoulder: CocoPart.RShoulder,
        #     MPIIPart.LShoulder: CocoPart.LShoulder,
        #     MPIIPart.LElbow: CocoPart.LElbow,
        #     MPIIPart.LWrist: CocoPart.LWrist,
        #     MPIIPart.Neck: CocoPart.Neck,
        #     MPIIPart.Nose: CocoPart.Nose,
        # }

        t = [
            (MPIIPart.Head, CocoPart.Nose),
            (MPIIPart.Neck, CocoPart.Neck),
            (MPIIPart.RShoulder, CocoPart.RShoulder),
            (MPIIPart.RElbow, CocoPart.RElbow),
            (MPIIPart.RWrist, CocoPart.RWrist),
            (MPIIPart.LShoulder, CocoPart.LShoulder),
            (MPIIPart.LElbow, CocoPart.LElbow),
            (MPIIPart.LWrist, CocoPart.LWrist),
            (MPIIPart.RHip, CocoPart.RHip),
            (MPIIPart.RKnee, CocoPart.RKnee),
            (MPIIPart.RAnkle, CocoPart.RAnkle),
            (MPIIPart.LHip, CocoPart.LHip),
            (MPIIPart.LKnee, CocoPart.LKnee),
            (MPIIPart.LAnkle, CocoPart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for mpi, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty


# CocoPairs = [
#     (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
#     (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
# ]   # = 19

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0)
]
# CocoPairsRender = CocoPairs[:-2]
CocoPairsRender = CocoPairs
# CocoPairsNetwork = [
#     (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#     (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
#  ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

parkinson_cut_list = [150, 254, 360, 528, 769, 919, 979, 1264, 1345, 1495, 1668, 1803, 2257, 2467, 2653, 2776, 2926,
                      2989, 3056, 3151,
                      3271, 3421, 3684, 4085, 4471, 4531, 4760, 4987, 5036, 5161, 5254, 5336, 5499, 5560, 5650, 5825,
                      6075, 6176,
                      6279, 6531, 6592, 6670, 6890, 7040, 7094, 7178, 7413, 7560, 7677, 7794, 7923, 7979, 8147, 8495,
                      8760, 8978,
                      9221, 9437, 9610, 9733, 9887, 9992, 10167, 10292, 10412, 10574, 10727, 10877, 10949, 11074, 11099,
                      11270,
                      11408, 11546, 11774, 12047, 12154, 12509, 12609, 12684, 12917, 13137, 13521, 13621, 13867, 13970,
                      14060,
                      14155, 14336, 14453, 14618, 14705, 14751, 14846, 14937, 15040, 15191, 15401, 15656, 15806, 15884,
                      16117,
                      16330, 16453, 16561, 16786, 16787, 17471, 17591]

cross_keypoints_list = [CrossPartCocoPoseTrack.Nose, None, None, None, None,
                        CrossPartCocoPoseTrack.LShoulder, CrossPartCocoPoseTrack.RShoulder,
                        CrossPartCocoPoseTrack.LElbow, CrossPartCocoPoseTrack.RElbow,
                        CrossPartCocoPoseTrack.LWrist, CrossPartCocoPoseTrack.RWrist,
                        CrossPartCocoPoseTrack.LHip, CrossPartCocoPoseTrack.RHip,
                        CrossPartCocoPoseTrack.LKnee, CrossPartCocoPoseTrack.RKnee,
                        CrossPartCocoPoseTrack.LAnkle, CrossPartCocoPoseTrack.RAnkle]


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(w, h):
    val_image = [
        read_imgfile('../images/p1.jpg', w, h),
        read_imgfile('../images/p2.jpg', w, h),
        read_imgfile('../images/p3.jpg', w, h),
        read_imgfile('../images/golf.jpg', w, h),
        read_imgfile('../images/hand1.jpg', w, h),
        read_imgfile('../images/hand2.jpg', w, h),
        read_imgfile('../images/apink1_crop.jpg', w, h),
        read_imgfile('../images/ski.jpg', w, h),
        read_imgfile('../images/apink2.jpg', w, h),
        read_imgfile('../images/apink3.jpg', w, h),
        read_imgfile('../images/handsup1.jpg', w, h),
        read_imgfile('../images/p3_dance.png', w, h),
    ]
    return val_image
