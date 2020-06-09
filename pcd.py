#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:37:55 2020

@author: abel
"""


import numpy as np
import cv2
from PIL import Image

fx = 2.771281292110203935e+02  # focal length x
fy = 2.897056274847714121e+02  # focal length y
cx = 1.600000000000000000e+02  # optical center x
cy = 1.200000000000000000e+02 # optical center y
scalingFactor = 1000.0


def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)
    
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()
    


if __name__=='__main__':
    verbose=True;
    with open('data/associate.txt') as file: 
        for line in file:
            contents=line.split()
            if verbose:
                print("File (Depth):",contents[0])
                print("File (RGB)",contents[1])
            break

    img_path=contents[0]
    depth_path=contents[1]

    
    generate_pointcloud(img_path,depth_path,'/home/ashfaquekp/sn-tsdf-fusion-python/pcd_sn10.ply')