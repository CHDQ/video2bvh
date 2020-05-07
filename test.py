from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton_bip
import numpy as np

cmu_skeleton_bip.CMUSkeleton().poses2bvh(np.load("./miscs/cxk_cache/3d_pose.npy"),
                                         output_file="./miscs/cxk_cache/biped.bvh")
