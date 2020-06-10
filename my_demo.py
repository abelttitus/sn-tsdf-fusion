#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:35:16 2020

@author: abel
"""


"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 300
  vol_bnds = np.array( [[-0.02348084,4.73584131],
 [ 0. ,2.70015462],
 [-3.38044459,1.0898702 ]]
)
  
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  cam_poses=np.loadtxt("data/camera-poses.txt")
  # file=open("data/associate.txt")
  # data = file.read()
  # lines = data.split("\n") 
  
  # i=0
  
  # for line in lines:                                     #This is used to loop all images
  #   contents=line.split(" ")
  #   try:
  #       depth_file=contents[1]
  #   except:
  #       print "Associate File read error at i =",i
  #       continue
    
  #   # Read depth image and camera pose
  #   depth_im = cv2.imread(depth_file,-1).astype(float)
  #   depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
  #   cam_pose=cam_poses[4*i:4*(i+1),:]

  #   # Compute camera view frustum and extend convex hull
  #   vol_bnds_temp=fusion.get_vol_bnds(depth_im,cam_pose)
  #   vol_bnds[:,0] = np.minimum(vol_bnds[:,0], vol_bnds_temp[:,0])
  #   vol_bnds[:,1] = np.maximum(vol_bnds[:,1], vol_bnds_temp[:,1])
  #   i+=1
  # file.close()
  

  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  
  file=open("data/associate.txt")
  data = file.read()
  lines = data.split("\n") 
  
  i=0
  
  for line in lines:                                     #This is used to loop all images
    contents=line.split(" ")
    try:
        depth_file=contents[1]
        rgb_file=contents[0]
    except:
        print "Associate File read error at i =",i
        continue
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_file,-1).astype(float)
    depth_im /= 1000.
    cam_pose=cam_poses[4*i:4*(i+1),:]

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    i+=1

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)