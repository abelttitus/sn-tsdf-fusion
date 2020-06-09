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
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  print "Camera Intrinsics",cam_intr
  cam_poses=np.loadtxt("data/camera-poses.txt")
  print "Cam poses shape",cam_poses.shape

  
  file=open("data/associate.txt")
  data = file.read()
  lines = data.split("\n") 
  
  vol_bnds = np.zeros((3,2))
  
  i=0
  for line in lines:     #This is used to loop all images
    contents=line.split(" ")
    rgb_file=contents[0]
    depth_file=contents[1]

    depth_im = cv2.imread(depth_file,-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    
    #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    #cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix
    cam_pose=cam_poses[4*i:4*(i+1),:]
    #print "Concatenated Cam pose",cam_pose
    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    i+=1
    
  file.close()  
  print "Volume Bounds:",vol_bnds


  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  file = open("data/associate.txt")
  data = file.read()
  lines = data.split("\n") 
  
  i=0
  for line in lines:     #This is used to loop all images
    contents=line.split(" ")
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    rgb_file=contents[0]
    depth_file=contents[1]
    cam_pose=cam_poses[4*i:4*(i+1),:]
    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_file,-1).astype(float)
    depth_im /= 1000.
    #depth_im[depth_im == 65.535] = 0
    
    
    i+=1
    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
  file.close()
  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh-sn10.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc-sn10.ply", point_cloud)
