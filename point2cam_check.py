# -*- coding: utf-8 -*

import os
import sys
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# import opencv as cv2
img = '/data/velo2cam/data_object_image_2/testing/6.jpeg'
binary = '/data/velo2cam/data_object_velodyne/testing/6.bin'
with open('./testing/calib/left.txt','r') as f:
    calib = f.readlines()


P = np.matrix([float(x) for x in calib[0].strip('\n').split(' ')[1:]]).reshape(3,3)
P = np.insert(P,3,values=[0,0,0],axis=1)
print("P is:",P)
d = np.matrix([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(1,5)
print("d is:",d)
R = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,3)
t = np.matrix([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,1)
print("R is:",R)
print("t is:",t)
# 3x4
Tr_velo_to_cam = np.hstack((R,t))
print("Tr_velo_to_cam.shape:",Tr_velo_to_cam.shape)

# 4x4
Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
print("Tr_velo_to_cam is:",Tr_velo_to_cam)

# 点云文件可能有x y z 三维  或者x y z intensity 四维信息
# scan = np.fromfile(binary, dtype=np.float32,count =-1).reshape([-1,4])
scan = np.fromfile(binary, dtype=np.float32,count =-1).reshape([-1,3])
points = scan[:, 0:3] # lidar xyz (front, left, up)
print("points.shape is",points.shape)
velo = np.insert(points,3,1,axis=1).T
print("1-velo.shape is",velo.shape)
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
# 3x4 * 4x4 * 4xn
cam = P  * Tr_velo_to_cam * velo #
print("cam shape is",cam.shape)

cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
print("after delete cam shape is",cam.shape)
cam[:2] /= cam[2,:]
print("after /= cam shape is",cam.shape)
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)



png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(P,d,(IMG_W,IMG_W),1, (IMG_W,IMG_W))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(png)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=0.5)
plt.title(6)
plt.savefig('./data_object_image_2/testing/6.png',bbox_inches='tight')
plt.show()