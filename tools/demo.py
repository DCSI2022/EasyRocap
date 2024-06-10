#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) 2024 -2024 WangHaoyu.All Rights Reserved
import argparse
from yolox.exp import get_exp
from loguru import logger
import torch
import os
import cv2
import time
import numpy as np
from yolox.utils import get_model_info
from Predictor_class import Predictor, VOC_CLASSES
from trackers.ocsort_tracker.ocsort1 import OCSort as OCSort1
from trackers.ocsort_tracker.ocsort2 import OCSort as OCSort2
from trackers.ocsort_tracker.ocsort3 import OCSort as OCSort3
from trackers.ocsort_tracker.ocsort4 import OCSort as OCSort4
from trackers.ocsort_tracker.ocsort5 import OCSort as OCSort5
from toolfunctions import read_cam_paras, batch_triangulate, initialize_annots,Get_Order_List, projectN3

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("-n", "--num_mark", default=3, type=int, help="number of markers")
    parser.add_argument("-m", "--mot_thre", default=0.6, type=float, help="the threshold of tracker")
    parser.add_argument(
        "--proj_thre",
        default=60,
        type=float,
        help="the threshold of projection error")
    parser.add_argument(
        "-r",
        "--path_root",
        default=None,
        type=str,
        help="the location of data")
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="weights_path")
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    return parser

def loadmodel(weights_file, Exp_file, conf, nms):
    # Parameters
    device = "gpu"
    legacy = False
    fp16 = False
    # ckpt for eval
    ckpt_file = weights_file
    exp_file = Exp_file
    exp = get_exp(exp_file)
    conf = conf
    # nms threshold
    nms = nms
    # img size
    tsize = 640
    if conf is not None:
        exp.test_conf = conf
    if nms is not None:
        exp.nmsthre = nms
    if tsize is not None:
        exp.test_size = (tsize, tsize)
    # initialize object model
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if device == "gpu":
        model.cuda()
    # Lock BatchNormalization and Dropout
    model.eval()
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    trt_file = None
    decoder = None
    predictor = Predictor(
        model, exp, VOC_CLASSES, trt_file, decoder,
        device, fp16, legacy,
    )
    return predictor

def trk2arr(num_balls,online_targets):
    frame_points = np.zeros((num_balls, 3))
    # cf = float(1)
    for t in online_targets:
        tid = t[4]
        # modify 2 store id midx midy
        j = int(tid) - 1
        mid_x = (t[0] + t[2]) / 2  # float
        mid_y = (t[1] + t[3]) / 2  # float
        frame_points[j, 0] = mid_x
        frame_points[j, 1] = mid_y
        frame_points[j, 2] = t[5] # confidence score
    return frame_points

def main(args):
    predictor = loadmodel(args.weights, args.exp_file, args.conf, args.nms)
    ptest_size = predictor.test_size
    num_mark = args.num_mark
    # build trackers for multi-view cameras
    tracker1 = OCSort1(det_thresh=args.mot_thre, num_objs=num_mark)
    tracker2 = OCSort2(det_thresh=args.mot_thre, num_objs=num_mark)
    tracker3 = OCSort3(det_thresh=args.mot_thre, num_objs=num_mark)
    tracker4 = OCSort4(det_thresh=args.mot_thre, num_objs=num_mark)
    tracker5 = OCSort5(det_thresh=args.mot_thre, num_objs=num_mark)
    # camera names
    path_root = args.path_root
    path_cams = os.path.join(path_root, "images")
    subfolders = sorted(os.listdir(path_cams))
    n_views = len(subfolders)
    camera_Pall = read_cam_paras(path_root, subfolders)
    # create a dict to store filename
    file_dict = {}
    index_list = Get_Order_List(n_views, num_mark)
    len_orders = len(index_list)
    # initial index_array
    arr_index = index_list[0]
    min_frames = 10000
    for subfolder in subfolders:
        subfolder_path = os.path.join(path_cams, subfolder)
        files = sorted(os.listdir(subfolder_path))
        num_frames = len(files)
        if num_frames < min_frames:
            min_frames = num_frames
        file_dict[subfolder] = files
        tmp_file = os.path.join(subfolder_path, files[0])
    test_img = cv2.imread(tmp_file)
    width = test_img.shape[1]
    height = test_img.shape[0]

    keys = list(file_dict.keys())
    for frame_id in range(min_frames):
        frames = []
        for key in keys:
            filenames_list = file_dict[key]
            filename = os.path.join(path_cams, key, filenames_list[frame_id])
            image_raw = cv2.imread(filename)
            frames.append(image_raw)
            file = filenames_list[frame_id]
        # frames:同时刻的多视角图像
        t0 = time.time()
        # input -> frame
        # output -> filtered markers position
        outputs, img_info = predictor.batch_inference(frames)
        online_targets1 = tracker1.update(outputs[0], [height, width], ptest_size)
        online_targets2 = tracker2.update(outputs[1], [height, width], ptest_size)
        online_targets3 = tracker3.update(outputs[2], [height, width], ptest_size)
        online_targets4 = tracker4.update(outputs[3], [height, width], ptest_size)
        online_targets5 = tracker5.update(outputs[4], [height, width], ptest_size)
        cam1 = trk2arr(num_mark, online_targets1)
        cam2 = trk2arr(num_mark, online_targets2)
        cam3 = trk2arr(num_mark, online_targets3)
        cam4 = trk2arr(num_mark, online_targets4)
        cam5 = trk2arr(num_mark, online_targets5)
        test_data = [cam1, cam2, cam3, cam4, cam5]
        keypoints2d = initialize_annots(test_data, n_views, num_mark, arr_index)
        keypoints3d = batch_triangulate(keypoints2d, camera_Pall)
        kptsRepro = projectN3(keypoints3d, camera_Pall)
        err = ((kptsRepro[:, :, 2] * keypoints2d[:, :, 2]) > 0.) * np.linalg.norm(
            kptsRepro[:, :, :2] - keypoints2d[:, :, :2], axis=2)
        sum_err = sum(sum(err))
        if sum_err < args.proj_thre:
            # print("Right index!")
            kps_eval = keypoints3d[:, :3]
            # print("The Repro_err is %10.3f" % sum_err)
        else:
            # search min_error
            min_err = 200
            for i in range(len_orders):
                keypoints2d = initialize_annots(test_data, n_views, num_mark, index_list[i])
                tmp_points3d = batch_triangulate(keypoints2d, camera_Pall)
                kptsRepro = projectN3(tmp_points3d, camera_Pall)
                err = ((kptsRepro[:, :, 2] * keypoints2d[:, :, 2]) > 0.) * np.linalg.norm(
                    kptsRepro[:, :, :2] - keypoints2d[:, :, :2], axis=2)
                sum_err = sum(sum(err))
                if sum_err < min_err:
                    min_err = sum_err
                    keypoints3d = tmp_points3d
                    arr_index = index_list[i]
            # print("ReCalculate Repro_error!")
            kps_eval = keypoints3d[:, :3]
            # print("The Repro_err is %10.3f" % min_err)
            # print("New array of index is:")
            # print(arr_index)
            # print("The Repro_err is %10.3f" % min_err)
        top_index = np.argmax(kps_eval[:, 2])
        position_3d = kps_eval[top_index, :]
        Fname = file[:-4]
        record_time = float(Fname)
        logger.info("The Robot's Position at time {:.3f} is {:.4f}, {:.4f}, {:.4f}".format(
            record_time, position_3d[0],position_3d[1],position_3d[2]))

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)