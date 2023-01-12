import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cv2
from tqdm import trange
from argparse import Namespace
import os
from tqdm import tqdm


def mono_pose(pred_path, gt_path):
    gt_txt = open(gt_path, 'r')

    poses = []

    for line in tqdm(gt_txt.readlines()):
        values = line.split()

        rot = [[float(values[0]), float(values[1]), float(values[2])], 
                    [float(values[4]), float(values[5]), float(values[6])],
                    [float(values[8]), float(values[9]), float(values[10])]]
        rot = R.from_matrix(rot).as_rotvec()
        tvec = [float(values[3]), float(values[7]), float(values[11])]

        poses.append({'gt':[rot.tolist(), np.array(tvec)]})


    pred_text = open(pred_path, 'r')
    for i, line in enumerate(tqdm(pred_text.readlines())):
        values = line.split()

        rot = [float(values[4]), float(values[5]), float(values[6]), float(values[7])]
        rot = R.from_quat(rot).as_rotvec()
        tvec = [float(values[1]), float(values[2]), float(values[3])]

        poses[i+1]['pred'] = [rot.tolist(), np.array(tvec)]

    poses[0]['pred'] = poses[0]['gt']
    
    return poses

def stereo_pose(pred_path, gt_path):
    gt_txt = open(gt_path, 'r')

    poses = []

    for line in tqdm(gt_txt.readlines()):
        values = line.split()

        rot = [[float(values[0]), float(values[1]), float(values[2])], 
                    [float(values[4]), float(values[5]), float(values[6])],
                    [float(values[8]), float(values[9]), float(values[10])]]
        rot = R.from_matrix(rot).as_rotvec()
        tvec = [float(values[3]), float(values[7]), float(values[11])]

        poses.append({'gt':[rot.tolist(), np.array(tvec)]})

    pred_text = open(pred_path, 'r')

    for i, line in enumerate(tqdm(pred_text.readlines())):
        values = line.split()

        rot = [[float(values[0]), float(values[1]), float(values[2])], 
                    [float(values[4]), float(values[5]), float(values[6])],
                    [float(values[8]), float(values[9]), float(values[10])]]
        rot = R.from_matrix(rot).as_rotvec()
        tvec = [float(values[3]), float(values[7]), float(values[11])]

        poses[i]['pred'] = [rot.tolist(), np.array(tvec)]

    return poses

def compute_error(poses):
    error_Rs = []
    error_Ts = []

    for pose in tqdm(poses[1:]):
        try:
            error_Ts.append(np.linalg.norm(pose['pred'][1] - pose['gt'][1]))

            norm_R = pose['pred'][0] / np.linalg.norm(pose['pred'][0])
            norm_gt_R = pose['gt'][0] / np.linalg.norm(pose['gt'][0])
            diff_R = np.clip(np.sum(norm_R * norm_gt_R), 0, 1)
            error_Rs.append(np.degrees(np.arccos(2 * diff_R * diff_R - 1)))
        except:
            pass

    return (np.median(error_Rs), np.median(error_Ts))

seq = '05'
gt_path = './kitti/data_odometry_poses/dataset/poses/%s.txt'%(seq)
mono_poses = mono_pose('/home/pinya/3dcv/ORB_SLAM3/result/mono/Mono_FrameTrajectory_%s.txt'%(seq), gt_path)
# stereo_poses = stereo_pose('/home/pinya/3dcv/ORB_SLAM3/result/our_warp/Stereo_FrameTrajectory_%s.txt'%(seq), gt_path)
stereo_poses_kitti = stereo_pose('/home/pinya/3dcv/ORB_SLAM3/result/kitti/Stereo_FrameTrajectory_%s.txt'%(seq), gt_path)
stereo_poses = stereo_pose('/home/pinya/3dcv/ORB_SLAM3/Stereo_FrameTrajectory_%s.txt'%(seq), gt_path)
# print(stereo_poses[0])

mono = compute_error(mono_poses)
stereo = compute_error(stereo_poses)
stereo_kitti = compute_error(stereo_poses_kitti)

print('mono : ', mono)
print('stereo : ', stereo)
print('stereo_kitti : ', stereo_kitti)