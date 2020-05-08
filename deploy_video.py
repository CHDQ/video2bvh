import sys

openpose_dir = '/space2/qiong.duan/openpose'
sys.path.append(f'{openpose_dir}/build/python/')  # openpose路径
from pose_estimator_2d import openpose_estimator
from pose_estimator_3d import estimator_3d
from utils import smooth, vis, camera
from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton

import cv2
import importlib
import numpy as np
import os
from pathlib import Path
from IPython.display import HTML

e2d = openpose_estimator.OpenPoseEstimator(
    model_folder=f'{openpose_dir}/models/')  # set model_folder to /path/to/openpose/models

video_file = Path('miscs/cxk.mp4')  # video file to process
output_dir = Path(f'miscs/{video_file.stem}_cache')
if not output_dir.exists():
    os.makedirs(output_dir)

cap = cv2.VideoCapture(str(video_file))
keypoints_list = []
img_width, img_height = None, None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_height = frame.shape[0]
    img_width = frame.shape[1]

    # returned shape will be (num_of_human, 25, 3)
    # last dimension includes (x, y, confidence)
    keypoints = e2d.estimate(img_list=[frame])[0]
    if not isinstance(keypoints, np.ndarray) or len(keypoints.shape) != 3:
        # failed to detect human
        keypoints_list.append(None)
    else:
        # we assume that the image only contains 1 person
        # multi-person video needs some extra processes like grouping
        # maybe we will implemented it in the future
        keypoints_list.append(keypoints[0])
cap.release()

# %% md

## Process 2D pose

# %%

# filter out failed result
keypoints_list = smooth.filter_missing_value(
    keypoints_list=keypoints_list,
    method='ignore'  # interpolation method will be implemented later
)

# smooth process will be implemented later

# save 2d pose result
pose2d = np.stack(keypoints_list)[:, :, :2]
pose2d_file = Path(output_dir / '2d_pose.npy')
np.save(pose2d_file, pose2d)

# %% md

## Visualize 2D pose

# %%

cap = cv2.VideoCapture(str(video_file))
vis_result_dir = output_dir / '2d_pose_vis'  # path to save the visualized images
if not vis_result_dir.exists():
    os.makedirs(vis_result_dir)

op_skel = openpose_skeleton.OpenPoseSkeleton()

for i, keypoints in enumerate(keypoints_list):
    ret, frame = cap.read()
    if not ret:
        break

    # keypoint whose detect confidence under kp_thresh will not be visualized
    vis.vis_2d_keypoints(
        keypoints=keypoints,
        img=frame,
        skeleton=op_skel,
        kp_thresh=0.4,
        output_file=vis_result_dir / f'{i:04d}.png'
    )
cap.release()

# %% md

## Initialize 3D pose estimator

# %%

importlib.reload(estimator_3d)
e3d = estimator_3d.Estimator3D(
    config_file='models/openpose_video_pose_243f/video_pose.yaml',
    checkpoint_file='models/openpose_video_pose_243f/best_58.58.pth'
)

# %% md

## Estimate 3D pose from 2D pose

# %%

pose2d = np.load(pose2d_file)
pose3d = e3d.estimate(pose2d, image_width=img_width, image_height=img_height)

# %% md

## Convert 3D pose from camera coordinates to world coordinates

# %%

subject = 'S1'
cam_id = '55011271'
cam_params = camera.load_camera_params('cameras.h5')[subject][cam_id]
R = cam_params['R']
T = 0
azimuth = cam_params['azimuth']

pose3d_world = camera.camera2world(pose=pose3d, R=R, T=T)
pose3d_world[:, :, 2] -= np.min(pose3d_world[:, :, 2])  # rebase the height

pose3d_file = output_dir / '3d_pose.npy'
np.save(pose3d_file, pose3d_world)

# %% md

## Visualize 3D pose

# %%

h36m_skel = h36m_skeleton.H36mSkeleton()
gif_file = output_dir / '3d_pose_300.gif'  # output format can be .gif or .mp4

ani = vis.vis_3d_keypoints_sequence(
    keypoints_sequence=pose3d_world[0:300],
    skeleton=h36m_skel,
    azimuth=azimuth,
    fps=60,
    output_file=gif_file
)
HTML(ani.to_jshtml())

# %% md

## Convert 3D pose to BVH

# %%

bvh_file = output_dir / f'{video_file.stem}.bvh'
cmu_skel = cmu_skeleton.CMUSkeleton()
channels, header = cmu_skel.poses2bvh(pose3d_world, output_file=bvh_file)

# %%

output = 'miscs/h36m_cxk.bvh'
h36m_skel = h36m_skeleton.H36mSkeleton()
_ = h36m_skel.poses2bvh(pose3d_world, output_file=output)
