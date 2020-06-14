#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:35:51 2020

@author: abel
"""
import time

import cv2
import numpy as np

import fusions


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
  class_id=40
  depth_dir='/home/ashfaquekp/val/0/10/depth/'
  mask_dir='/home/ashfaquekp/val/0/10/mrcnn_mask/'
  file=open("/home/ashfaquekp/val/0/10/mrcnn_mask/class_"+str(class_id)+".txt")
  data = file.read()
  lines = data.split("\n") 
  
  i=0
  
  for line in lines:                                     #This is used to loop all images
    contents=line.split(" ")
    try:
      rgb_file=contents[0]

      img_no=rgb_file.split(".")[0].split("/")[-1]
      depth_file=depth_dir+str(img_no)+'.png'
      instance_mask_file=mask_dir+str(img_no)+'_'+str(class_id)+'.jpg'
    
      depth_im = cv2.imread(depth_file,-1).astype(float)
      depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    
      mask=cv2.imread(instance_mask_file)
      mask=mask.astype('float')
      mask[mask<125]=0.0
      mask[mask>=125]=1.0
    
      depth_im=depth_im*mask[:,:,0]
      
      pose_index=int(int(img_no)/25)
      cam_pose=cam_poses[4*pose_index:4*(pose_index+1),:]
  
# Compute camera view frustum and extend convex hull
      vol_bnds_temp=fusions.get_vol_bnds_obj(depth_im,cam_pose)
      vol_bnds[:,0] = np.minimum(vol_bnds[:,0], vol_bnds_temp[:,0])
      vol_bnds[:,1] = np.maximum(vol_bnds[:,1], vol_bnds_temp[:,1])
    except:
      print "Associate File read error at i =",i
      continue

    i+=1

    
  file.close()
  

  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusions.TSDFVolume(vol_bnds, voxel_size=0.001)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  
  # class_id=40
  # depth_dir='/home/ashfaquekp/val/0/10/depth/'
  # mask_dir='/home/ashfaquekp/val/0/10/mrcnn_mask/'
  file=open("/home/ashfaquekp/val/0/10/mrcnn_mask/class_"+str(class_id)+".txt")
  data = file.read()
  lines = data.split("\n") 
  
  i=0
  
  for line in lines:                                     #This is used to loop all images
    contents=line.split(" ")
    try:
        rgb_file=contents[0] 
        img_no=rgb_file.split(".")[0].split("/")[-1]
        depth_file=depth_dir+str(img_no)+'.png'
        instance_mask_file=mask_dir+str(img_no)+'_'+str(class_id)+'.jpg'
        print("Fusing frame %d"%(i+1))
        
        pose_index=int(int(img_no)/25)
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_file,-1).astype(float)
        depth_im /= 1000.
        cam_pose=cam_poses[4*pose_index:4*(pose_index+1),:]
    
        mask=cv2.imread(instance_mask_file)
        mask=mask.astype('float')
        mask[mask<125]=0.0
        mask[mask>=125]=1.0
        
        color_image=color_image*mask
        color_image=color_image.astype('int')
        
        depth_im=depth_im*mask[:,:,0]
        
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    except:
        print "Associate File read error at i =",i
        continue
    i+=1

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusions.meshwrite("mesh-bottle-obj-vol.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  # print("Saving point cloud to pc.ply...")
  # point_cloud = tsdf_vol.get_point_cloud()
  # fusions.pcwrite("pc-key.ply", point_cloud)