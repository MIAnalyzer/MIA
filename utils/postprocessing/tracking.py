# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:56:31 2020

@author: Koerber
"""

#https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
# hungarian algorithm

from scipy.optimize import linear_sum_assignment
import numpy as np

class ObjectTracking():
    def __init__(self,parent):
        self.parent = parent
        self.tracking_list = []
        self.thresh = 100
        self.fadeawayperiod = 1
        self.newobjectperiod = 2
        self.objects = None
        self.timepoints = 0
        # results
        self.tracks = []
        
    def addObject(self, obj, timepoint):
        pass
    
    def deleteObject(self, obj, timepoint=None):
        if timepoint is  None:
            # delete for all timepoints
            pass 
        
    def getObjectTrack(self, objnum):
        if len(self.tracks)>objnum and objnum > 0:   
            return self.tracks[objnum-1]
        else:
            return None

    def performTracking(self, sequence):
        self.tracks = []
        self.timepoints = len(sequence)
        for tp,t in enumerate(sequence):
            shapes = self.parent.dl.Mode.LoadShapes(t)
            self.addTimePoint(self.parent.dl.Mode.LoadShapes(t), tp)
            for i in range(self.objects.shape[0]):
                if len(self.tracks) < i+1:
                    self.tracks.append([(-1,-1)]*tp)
                if self.objects[i,tp] > 0:
                    shapes[self.objects[i,tp]-1].setObjectNumber(i+1) 
                    self.tracks[i].append(shapes[self.objects[i,tp]-1].getPosition())
                else:
                    self.tracks[i].append((-1,-1))
            self.parent.dl.Mode.saveShapes(shapes, t)


    
    def addTimePoint(self, detections, t):
        if t == 0:
            self.objects = np.zeros((len(detections),self.timepoints),dtype=int)
            self.objects[:,0] = range(1,len(detections)+1)
            self.tracking_list = [(x,i) for (x,i) in zip(detections,range(1,len(detections)+1))]
        else:
            self.calculateMatches( detections, t)
        
    def calculateMatches(self,  t, tp):
        t_minus_one = [x for (x,i) in self.tracking_list]
        corr_mat = [self.parent.dl.Mode.LabelDistance(x,y) for x in t_minus_one for y in t]
        mat = np.asarray(corr_mat).reshape(len(t_minus_one),len(t))
        row,col = linear_sum_assignment(mat)


        # matched     
        for r,c in zip(row,col):
            if mat[r,c] < self.thresh and t_minus_one[r].classlabel == t[c].classlabel:
                pos = self.tracking_list[r][1]-1
                self.objects[pos,tp] = c+1
            else:
                newobject = np.zeros((1,self.timepoints),dtype=int)
                newobject[0,tp] = c+1
                self.objects = np.append(self.objects,newobject, axis=0)
                self.tracking_list.append((t[c], self.objects.shape[0]))
                
        # unmatched
        for track in range(len(t)):
            if track not in col:
                newobject = np.zeros((1,self.timepoints),dtype=int)
                newobject[0,tp] = track+1
                self.objects = np.append(self.objects,newobject, axis=0)
                self.tracking_list.append((t[track], self.objects.shape[0]))
           

        # remove continuously undetected 
        gone = []
        if tp-self.fadeawayperiod >= 0:
            for x in self.tracking_list:
                if self.objects[x[1]-1,tp] == 0:
                    occ = self.objects[x[1]-1,tp-self.fadeawayperiod:tp]
                    if np.sum(occ) == 0:
                        gone.append(x)    
            for obj in gone:
                self.tracking_list.remove(obj)

                

