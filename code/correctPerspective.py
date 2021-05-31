#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 08:39:13 2021

@author: nuriamartinezbarnola
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import *
from scipy import ndimage
import cv2

def correctPerspective(image):
    imag=prepar(image)
    image=imag[0]
    w=imag[1]
    h=imag[2]
    im_org=imag[3]
    pixel=Image.getpixel(image,(0,0))
    if type(pixel)==int:
        im=image
    else:
        width,height=image.size
        image_array=np.asarray(image)
        im=np.zeros((height,width))
        for i in range(height):
            for j in range(width):
                t=image_array[i, j]
                ts=sum(t)/len(t)
                im[i, j]=ts           
    #data=image.getdata()
    #print(data)
    im_array=np.asarray(im, dtype=np.float64)
    #print(im_array.size)
    ix=ndimage.sobel(im_array, 0)
    iy=ndimage.sobel(im_array, 1)
    ix2=ix*ix
    iy2=iy*iy
    ixy=ix*iy
    ix2=ndimage.gaussian_filter(ix2, sigma=2)
    iy2=ndimage.gaussian_filter(iy2, sigma=2)
    ixy=ndimage.gaussian_filter(ixy, sigma=2)
    w,h=im_array.shape
    res=np.zeros((w,h))
    im_r=np.zeros((w,h))
    max_r=0
    for i in range(w):
        for j in range(h):
            mat=np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            res[i, j] = np.linalg.det(mat) - 0.15*(np.power(np.trace(mat), 2))
            if res[i, j] > max_r:
                max_r = res[i, j]
                
    for i in range(w-1):
        for j in range(h-1):
            if (res[i, j] > 0.01*max_r) and (res[i, j] > res[i-1, j-1]) and (res[i, j] > res[i-1, j+1]) and (res[i, j] > res[i+1, j-1]) and (res[i, j] > res[i+1, j+1]):
                im_r[i, j] = 1
    
    pc, pr = np.where(im_r == 1)
    #plt.plot(pr, pc, 'r+')
    #plt.imshow(im, 'gray')
    #plt.show()
    #return (pr,pc)
    show([pr,pc],im_org,w,h)

def prepar(image):
    im_org=image
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,3)
    w=t.shape[0]
    h=t.shape[1]
    cnts=cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    for c in cnts:
        area=cv2.contourArea(c)
        if (area<1000):
            cv2.drawContours(t,[c],-1,(0,0,0),-1)
    #plt.imshow(t,'gray')
    t=fromarray(t)
    return (t,w,h,im_org)
#p=detect(t)
def show(p,image,w,h):
    pts=[]
    for count,i in enumerate(p[0]):
        pts.append([i])
    for count,j in enumerate(p[1]):
        pts[count].append(j)
    
    two=False
    points=[]
    for pt in pts:
        if two:
            if pt[0]>ant[0]-50 and pt[0]<ant[0]+50 and pt[0]>ant[0]-170 and pt[0]<ant[0]+170:
                same=True
            else:
                points.append(pt)
        else:
            points.append(pt)
            two=True
        ant=pt
    llist=[]
    coord=[[0, 0], [h, 0],[0, w],[h, w]]
    coord2=[[0, 0], ['h', 0],[0, 'w'],['h', 'w']]
    
    for pt in points:
        dist_min=100000
        c_min=0
        for cont,x in enumerate(coord):
            distancia1 = math.sqrt((x[0]-pt[0])**2+(x[1]-pt[1])**2)
            if distancia1<dist_min:
                dist_min=distancia1
                c_min=coord2[cont]
        llist.append([pt,c_min])
        
    
    p_ord=[]  
    for cont,co in enumerate(coord2):
        for x in llist:
            if x[1]==co:
                p_ord.append(x[0])
    
    pts1 = np.float32(p_ord) 
    pts2 = np.float32([[0, 0], [h, 0],[0, w],[h, w]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(h,w))
    plt.imshow(dst)
    plt.show()
    
    
    
image=cv2.imread('1.png')
correctPerspective(image) 

