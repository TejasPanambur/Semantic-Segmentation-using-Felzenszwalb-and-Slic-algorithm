# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:50:08 2019

@author: TEJAS
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation as seg
import cv2
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.measure import regionprops
from skimage import draw
from plot_rag_merge import _weight_mean_color, merge_mean_color


class computeLibs:
    def felzenszwalb(self,felzParam,img,plot=True,save=0):
        colorspace = felzParam['colorSpace_s']
        felzenszwalb_img = seg.felzenszwalb(img,scale=felzParam['scale'],sigma=felzParam['sigma'],min_size=felzParam['min_size'])
        if plot:
            if colorspace=='hsv':
                img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
            plotLibs().plotBoundaries(img,felzenszwalb_img,save=save,title='F-H segmented Image')
        return felzenszwalb_img
    
    def slic(self,slicParam,img,plot=True,save=0):
        colorspace = slicParam['colorSpace_s']
        slic_img = seg.slic(img, n_segments=slicParam['n_segments'], compactness=slicParam['compactness'], max_iter=slicParam['max_iter'])
        if plot:
            if colorspace=='lab':
                img = cv2.cvtColor(img,cv2.COLOR_Lab2BGR)
            plotLibs().plotBoundaries(img,slic_img,save=save,title='Slic Segmented Image')
        return slic_img
    
    def slic0(self,slic0Param,img,plot=True,save=0):
        colorspace = slic0Param['colorSpace_s']
        slic0_img = seg.slic(img,n_segments=slic0Param['n_segments'],max_iter=slic0Param['max_iter'],slic_zero=True)
        if plot:
            if colorspace=='lab':
                img = cv2.cvtColor(img,cv2.COLOR_Lab2BGR)
            plotLibs().plotBoundaries(img,slic0_img,save=save,title='Slic0 segmented Image')
        return slic0_img
    
    def labels2img(self,img,labels,plot=True,save=0):
        labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
        label_rgb = color.label2rgb(labels, img, kind='avg')
        if plot:
            plotLibs().plotBoundaries(label_rgb,labels,save=save,title='Label rgb Image')
        return label_rgb
    
    def RAG(self,img, labels,plot=True,save=0):
        labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
        regions = regionprops(labels)
        rag = graph.rag_mean_color(img, labels)
        for region in regions:
            rag.node[region['label']]['centroid'] = region['centroid']
        if plot:  
            plotLibs().plotRAG(img,rag,save=save,title='RAG Image') 
        return rag
    
    def graphMergeHierarchical(self,img, labels,thresh=0.15,plot=True,save=0):
        g = graph.rag_mean_color(img,labels)
        new_labels = graph.merge_hierarchical(labels, g, thresh, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)
        if plot:        
            plotLibs().dispImg(color.label2rgb(new_labels, img, kind='avg'),save=save,title='Merge Hierarchical Label rgb Image')
            plotLibs().plotBoundaries(img,new_labels,save=save,title='Merge Hierarchical')
        return new_labels
    
    def graphNormalizedCuts(self,img, labels,thresh=0.5,num_cuts=100,plot=True,save=0):
        g = graph.rag_mean_color(img, labels)
        new_labels = graph.cut_normalized(labels, g,thresh,num_cuts)
        if plot:
            plotLibs().dispImg(color.label2rgb(new_labels, img, kind='avg'),save=save,title='Ncut Label rgb Image')
            plotLibs().plotBoundaries(img,new_labels,save=save,title='Ncut Boundary Images')
        return new_labels



class plotLibs:
    
    def dispImg(self,img,save=1,title='image'):
        plt.figure(figsize=(10,10))
        plt.title(title)
        plt.imshow(img)
        if save:
                plt.savefig('Output/'+'.jpg')            
        plt.show()
    
    def plot_3d(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img)
        r,g,b =  r.flatten(), g.flatten(), b.flatten()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()
        
    def __segments(self,img,labels,center):
        labels = np.array(labels,dtype='float32')
        labels[labels!=center]= np.nan
        labels[labels==center]=1
        labels[labels==np.nan] = 0
        return img*labels
        
    def dispSegment(self,img,labels,number_of_clusters,name):
        M,N = number_of_clusters//3,3
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(number_of_clusters):
            segment = self.__segments(img,labels,i)
            axs[i].imshow(segment)
            axs[i].set_title('segment'+str(i))  
        plt.savefig('Plots/Segments/'+name+str(number_of_clusters)+'.jpg')
        
    def dispKmeansBruteImg(self,reconstructedImg,l,plt_name):
        M,N = len(reconstructedImg)//2,2
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(len(reconstructedImg)):
            axs[i].imshow(reconstructedImg[i])
            axs[i].set_title('K_'+str(i+l))  
        plt.savefig('Plots/Kmeans/'+plt_name+'.jpg')
        plt.show()
        
   
    
    def __display_edges(self,image, g, threshold):
        """Draw edges of a RAG on its image
     
        Returns a modified image with the edges drawn.Edges are drawn in green
        and nodes are drawn in yellow.
     
        Parameters
        ----------
        image : ndarray
            The image to be drawn on.
        g : RAG
            The Region Adjacency Graph.
        threshold : float
            Only edges in `g` below `threshold` are drawn.
     
        Returns:
        out: ndarray
            Image with the edges drawn.
        """
        image = image.copy()
        for edge in g.edges:
            n1, n2 = edge
     
            r1, c1 = map(int, g.node[n1]['centroid'])
            r2, c2 = map(int, g.node[n2]['centroid'])
     
            line  = draw.line(r1, c1, r2, c2)
            circle = draw.circle(r1,c1,2)
     
            if g[n1][n2]['weight'] < threshold :
                image[line] = 0,1,0
            image[circle] = 1,1,0
     
        return image
    
    def plotRAG(self,img,rag,save=0,title='RAG Image'):
        edges_drawn_all = self.__display_edges(img, rag, np.inf)
        self.dispImg(edges_drawn_all,save,title)   
        
    def plotRagwithColorMaps(self,img,labels):
        g = graph.rag_mean_color(img, labels)
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

        ax[0].set_title('RAG drawn with default settings')
        lc = graph.show_rag(labels, g, img, ax=ax[0])
        # specify the fraction of the plot area that will be used to draw the colorbar
        fig.colorbar(lc, fraction=0.03, ax=ax[0])
        
        ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
        lc = graph.show_rag(labels, g, img,
                            img_cmap='gray', edge_cmap='viridis', ax=ax[1])
        fig.colorbar(lc, fraction=0.03, ax=ax[1])
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def plotBoundaries(self,img,labels,save=0,title='Boundaries Image'):
        Boundary_Img = seg.mark_boundaries(img, labels)
        self.dispImg(Boundary_Img,save,title)            
                    
    
class imgLibs:
    
    def __init__(self,imgName,clrSpace='rgb'):
        self.img = np.load(imgName)
        self.imgShape = self.img.shape
        self.clrSpace = clrSpace
    
    def loadImg(self):
        self.img[np.isnan(self.img)] = 0
        if self.clrSpace =='rgb':
            return self.img
        elif self.clrSpace == 'hsv':
            return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        elif self.clrSpace == 'gray':
            return cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        elif self.clrSpace == 'lab':
            return cv2.cvtColor(self.img,cv2.COLOR_BGR2Lab)
        