# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:56:31 2020

@author: Koerber
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from utils.Image import readNumOfImageFrames



class ObjectTracking():
    def __init__(self,dl,files):
        self.dl = dl
        self.files = files
        self.tracking_list = []
        self.thresh = 35
        self.fadeawayperiod = 1
        self.newobjectperiod = 2
        self.objects = None
        self.timepoints = 0
        self.sequence = []
        self.stackMode = False
        # results
        self.tracks = None
          
    def deleteObject(self, obj):
        objnum = obj.objectNumber
        for tp,t in self.labelGenerator():
            shapes = self.dl.Mode.LoadShapes(t)
            numbers = [x.objectNumber for x in shapes]
            if objnum in numbers:
                shapes.remove(shapes[numbers.index(objnum)])
                self.dl.Mode.saveShapes(shapes, t)
        np.delete(self.tracks, objnum-1,0)

    def changeTimePoint(self, objects, tp):
        self.tracks[:,tp,:] = -1
        unmatched = []
        for o in objects.shapes:
            if o.objectNumber == -1 or self.tracks.shape[0] < o.objectNumber:
                unmatched.append(o)
            else:
                self.tracks[o.objectNumber-1,tp,:] = np.asarray(o.getPosition())
        if len(unmatched) == 0:
            return

        if tp != 0:
            t_minus_one = self.tracks[:,tp-1,:]
            t = self.tracks[:,tp,:]
            lost = np.where((t_minus_one > 0) & (t < 0))
            if lost[0].size == 0:
                for o in unmatched:
                    self.setObjectNumber(o,tp)
            else:
                pos = [x for x in zip(t_minus_one[lost][0::2],t_minus_one[lost][1::2])]
                corr_mat = [self.distance(x.getPosition(),y) for x in unmatched for y in pos]
                mat = np.asarray(corr_mat).reshape(len(unmatched),len(lost[0])//2)
                row,col = linear_sum_assignment(mat)
                for r,c in zip(row,col):
                    if self.stackMode:
                        frames = self.files.ImagePath2LabelPath(self.sequence[0], False, True)
                        prev_shapes = self.dl.Mode.LoadShapes(frames[[self.files.getFrameNumber(x) for x in frames].index(tp-1)])
                    else:
                        prev_shapes = self.dl.Mode.LoadShapes(self.files.ImagePath2LabelPath(self.sequence[tp-1]))
                    prev_shape = prev_shapes[[x.objectNumber for x in prev_shapes].index(lost[0][c*2]+1)]
                    if mat[r,c] < self.thresh and unmatched[r].classlabel == prev_shape.classlabel:
                        self.setObjectNumber(unmatched[r],tp,lost[0][c*2] + 1)
                    else:
                        self.setObjectNumber(unmatched[r],tp)

                for o in range(len(unmatched)):
                    if o not in row:
                        self.setObjectNumber(unmatched[o],tp)
        else:
            for o in unmatched:
                self.setObjectNumber(o,tp)

    def labelGenerator(self):
        if not self.files.files:
            return
        for tp in range(len(self.files.files)):
            imagefile = self.sequence[tp]
            t = self.files.ImagePath2LabelPath(imagefile, True, self.stackMode)
            if not t:
                continue
            if self.stackMode:
                for t_ in t:
                    yield tp + self.files.getFrameNumber(t_), t_
            else:
                yield tp, t

    def setObjectNumber(self, obj, tp=None, objnum=None):
        if objnum is None:
            objnum = self.getFirstFreeObjectNumber()
        if tp is None:
            tp = self.files.currentImage
        obj.objectNumber = objnum
        self.addTrackPosition(tp, obj)

    def getFirstFreeObjectNumber(self):
        free = np.where(np.all(self.tracks<0,axis=1))[0]
        if len(free) > 0:
            return free[0] + 1
        else:
            return self.tracks.shape[0]

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def getObjectTrack(self, objnum):
        if self.tracks.shape[0]>=objnum and objnum > 0:   
            return self.tracks[objnum-1,...].tolist()
        else:
            return None

    def getObjectOccurences(self, objnum):
        if self.tracks.shape[0]>=objnum and objnum > 0:   
            return np.where(self.tracks[objnum-1,...,0]>0)[0].tolist()
        else:
            return []

    def getTimePointFromImageName(self, imagepath):
        return self.sequence.index(imagepath)

    def getNumbersOfTrackedObjects(self):
        return np.where(np.any(self.tracks[...,0]>=0,axis=1))[0]+1

    def resetTracking(self, numofTimePoints, numofObjects=250):
        self.tracks = np.zeros((numofObjects,numofTimePoints,2),dtype=int)-1
        self.objects = None
        self.tracking_list = []

    def loadTrack(self):
        # load from files
        if not self.files.files:
            return
        self.sequence = self.files.files
        if len(self.files.files) > 1:
            self.timepoints = len(self.files.files)
            self.stackMode = False
        elif len(self.files.files) == 1:
            self.timepoints = readNumOfImageFrames(self.files.files[0])
            self.stackMode = True

        self.resetTracking(self.timepoints)
        for tp,t in self.labelGenerator():
            shapes = self.dl.Mode.LoadShapes(t)
            save = False
            for i,s in enumerate(shapes):
                # fix object number
                # that would preferably done with shape object 
                if s.objectNumber == -1:
                    s.objectNumber = i+1
                    save = True
                self.addTrackPosition(tp,s)
            if save:
                self.dl.Mode.saveShapes(shapes, t)
            self.fillBlanksForMissingObjects(tp)

    
    #def saveTrack(self, filename):
    # could be saved for performance reasons instead of reloaded from individual contour files
    #    pass

    def addTrackPosition(self, tp, obj):
        objnum = obj.objectNumber
        if self.tracks.shape[0] < objnum+1:
            self.tracks = np.append(self.tracks,np.zeros((objnum - self.tracks.shape[0],self.tracks.shape[1],2))-1,axis = 0)
        pos = obj.getPosition()
        self.tracks[objnum-1,tp,:] = np.asarray(pos)


    def fillBlanksForMissingObjects(self, tp):      
        [x.append((-1,-1)) for x in self.tracks if len(x) < tp+1]


    def performTracking(self):
        self.resetTracking(self.timepoints)
        for tp,t in self.labelGenerator():
            shapes = self.dl.Mode.LoadShapes(t)
            self.addTimePoint(self.dl.Mode.LoadShapes(t), tp)
            for i in range(self.objects.shape[0]):
                if self.objects[i,tp] > 0:
                    shapes[self.objects[i,tp]-1].setObjectNumber(i+1) 
                    self.addTrackPosition(tp,shapes[self.objects[i,tp]-1])
            self.fillBlanksForMissingObjects(tp)
            self.dl.Mode.saveShapes(shapes, t)

    def addTimePoint(self, detections, t):
        #if t == 0:
        if self.objects is None or self.tracking_list == []:
            self.objects = np.zeros((len(detections),self.timepoints),dtype=int)
            self.objects[:,0] = range(1,len(detections)+1)
            self.tracking_list = [(x,i) for (x,i) in zip(detections,range(1,len(detections)+1))]
        else:
            self.calculateMatches(detections, t)
            

    def calculateMatches(self,  t, tp):
        t_minus_one = [x for (x,i) in self.tracking_list]
        corr_mat = [self.dl.Mode.LabelDistance(x,y) for x in t_minus_one for y in t]
        mat = np.asarray(corr_mat).reshape(len(t_minus_one),len(t))          
         
        # improves performance because it avoid (or equalizes to be more precisely) matches larger self.thresh
        mat[mat>self.thresh] = 100000
        row,col = linear_sum_assignment(mat)

        # matched     
        for r,c in zip(row,col):
            if mat[r,c] < self.thresh and t_minus_one[r].classlabel == t[c].classlabel: 
                pos = self.tracking_list[r][1]-1
                # update contour
                self.tracking_list[r] = (t[c], self.tracking_list[r][1])
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

        

