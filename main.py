# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:01:01 2019

@author: TEJAS
"""

'''
    In this project, the goal is to segment rocks from soil. You will compare 3 
    methods for segmenting the images that you used in Project 1. 
1)	Felzenszwalb’s algorithm on its own with parameters set 
    so that to obtain the final segmentation into rock and soil
2)	Felzenszwalb’s algorithm with parameters set so that an 
    oversegmentation is obtained. The Region adjacency graph (RAG) is built on the 
    segments and hierarchical merging is used on the RAG to obtain the final segmentation into rock and Soil. 
3)	SLIC superpixels (over)segmentation. The Region adjacency graph (RAG) is 
    built on the segments and hierarichical merging is used on the RAG to obtain t
    he final segmentation into rock and Soil. 
'''


import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation as seg
from Functions import imgLibs,plotLibs,computeLibs
import cv2
from skimage.future import graph


###############################################################################
'''Switches to Hyper Parameters To play with'''
imageNum = 1 # 0,1,2,3 numbers for 4 images
F_H = 1  #1 to run FH algorithm 
slic = 1 #1 to run slic algorithm 
slic0 = 1 #1 to run slic0 algorithm 
plot_Rag = 1 #1 to plot RAG  
merge_hierarchical = 1 #1 to perform hierarchical merging
ncut =  1 #1 to perform ncuts
path = 'Dataset/'
images = [{'ImageName':'0153MR0008490000201265E01_DRLX',
           'felzenszwalb_params':{'colorSpace_s':'rgb','scale':100,'sigma':0.4,'min_size':400,'hcThresh':0.08,'ncuts':{'tresh':0.35,'num_cuts':100}},
           'slic_params':{'colorSpace_s':'lab','n_segments':160, 'compactness':500.0, 'max_iter':1000,'hcThresh':0.12,'ncuts':{'tresh':0.5,'num_cuts':100}},
           'slic0_params':{'colorSpace_s':'lab','n_segments':200,'max_iter':1000,'hcThresh':0.12,'ncuts':{'tresh':0.15,'num_cuts':80}}},
          {'ImageName':'0172ML0009240000104879E01_DRLX',
           'felzenszwalb_params':{'colorSpace_s':'rgb','scale':200,'sigma':0.5,'min_size':200,'hcThresh':0.08,'ncuts':{'tresh':0.5,'num_cuts':200}},
           'slic_params':{'colorSpace_s':'lab','n_segments':200, 'compactness':700.0, 'max_iter':1000,'hcThresh':0.07,'ncuts':{'tresh':0.2,'num_cuts':10}},
           'slic0_params':{'colorSpace_s':'lab','n_segments':160,'max_iter':1000,'hcThresh':0.07,'ncuts':{'tresh':0.1,'num_cuts':100}}},
          {'ImageName':'0172ML0009240340104913E01_DRLX',
           'felzenszwalb_params':{'colorSpace_s':'rgb','scale':100,'sigma':0.9,'min_size':200,'hcThresh':0.1,'ncuts':{'tresh':0.14,'num_cuts':100}},
          'slic_params':{'colorSpace_s':'lab','n_segments':140, 'compactness':700.0, 'max_iter':1000,'hcThresh':0.08,'ncuts':{'tresh':0.2,'num_cuts':100}},
          'slic0_params':{'colorSpace_s':'lab','n_segments':200,'max_iter':1000,'hcThresh':0.05,'ncuts':{'tresh':0.2,'num_cuts':100}}},
          {'ImageName':'0270MR0011860360203259E01_DRLX',
           'felzenszwalb_params':{'colorSpace_s':'rgb','scale':150,'sigma':0.9,'min_size':100,'hcThresh':0.1,'ncuts':{'tresh':0.075,'num_cuts':100}},
          'slic_params':{'colorSpace_s':'lab','n_segments':250, 'compactness':900.0, 'max_iter':1000,'hcThresh':0.08,'ncuts':{'tresh':0.2,'num_cuts':100}},
          'slic0_params':{'colorSpace_s':'lab','n_segments':250,'max_iter':1000,'hcThresh':0.1,'ncuts':{'tresh':0.09,'num_cuts':100}}}]

###############################################################################
imageName = images[imageNum]['ImageName']

def loadImgHelper(fpath,colorSpace_s,plot=True):
    colorSpace_img = imgLibs(fpath+'.npy',colorSpace_s).loadImg()
    if plot:
        if colorSpace_s=='lab':
            img = cv2.cvtColor(colorSpace_img,cv2.COLOR_Lab2BGR)
        if colorSpace_s=='hsv':
            img = cv2.cvtColor(colorSpace_img,cv2.COLOR_HSV2BGR)
        if colorSpace_s == 'rgb':
            img = colorSpace_img
        plotLibs().dispImg(img,save=0,title='Input Image')
    return img,colorSpace_img

def ragHelper(img,labels):
    computeLibs().RAG(img,labels)
    computeLibs().labels2img(img,labels)
    plotLibs().plotRagwithColorMaps(img,labels)

if F_H:
    print('Running FH Algorithm...')
    felzParam =  images[imageNum]['felzenszwalb_params']
    img,colorSpace_img = loadImgHelper(path+imageName,felzParam['colorSpace_s'])
    labels = computeLibs().felzenszwalb(felzParam,colorSpace_img)
    if plot_Rag:
        print('Computing RAG.....')
        ragHelper(img,labels)        
    if merge_hierarchical:
        print('Computing hierarchical merge.....')
        merge_hierarch = felzParam['hcThresh']
        new_labels = computeLibs().graphMergeHierarchical(img,labels,thresh=merge_hierarch)
    if ncut:
        print('Computing Normalized Cuts.....')
        ncut_tresh,num_cuts = felzParam['ncuts']['tresh'],felzParam['ncuts']['num_cuts']
        new_labels = computeLibs().graphNormalizedCuts(img,labels,thresh=ncut_tresh,num_cuts=num_cuts)

    
if slic:
    print('Running SLIC Algorithm...')
    slicParam =  images[imageNum]['slic_params']
    img,colorSpace_img = loadImgHelper(path+imageName,slicParam['colorSpace_s'])
    labels = computeLibs().slic(slicParam,colorSpace_img)
    if plot_Rag:
        print('Computing RAG.....')
        ragHelper(img,labels)        
    if merge_hierarchical:
        print('Computing hierarchical merge.....')
        merge_hierarch = slicParam['hcThresh']        
        new_labels = computeLibs().graphMergeHierarchical(img,labels,thresh=merge_hierarch)
    if ncut:
        print('Computing Normalized Cuts.....')
        ncut_tresh,num_cuts = slicParam['ncuts']['tresh'],slicParam['ncuts']['num_cuts']
        new_labels = computeLibs().graphNormalizedCuts(img,labels,thresh=ncut_tresh,num_cuts=num_cuts)

    
if slic0:
    print('Running SLIC0 Algorithm...')
    slic0Param =  images[imageNum]['slic0_params']
    img,colorSpace_img = loadImgHelper(path+imageName,slic0Param['colorSpace_s'])
    labels = computeLibs().slic0(slic0Param,colorSpace_img)
    if plot_Rag:
        print('Computing RAG.....')
        ragHelper(img,labels)        
    if merge_hierarchical:
        print('Computing hierarchical merge.....')
        merge_hierarch = slic0Param['hcThresh']
        new_labels = computeLibs().graphMergeHierarchical(img,labels,thresh=merge_hierarch)
    if ncut:
        print('Computing Normalized Cuts.....')
        ncut_tresh,num_cuts = slic0Param['ncuts']['tresh'],slic0Param['ncuts']['num_cuts']
        new_labels = computeLibs().graphNormalizedCuts(img,labels,thresh=ncut_tresh,num_cuts=num_cuts)


    







    
    



