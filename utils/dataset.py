#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:41:22 2018

@author: Caroline Pacheco do E.Silva
"""
#from __future__ import print_function, division
import os
import cv2
import glob
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader


face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

class FaceDetection(Dataset):
    """Face Detection Dataset 
    Arguments:
        dataset_path (string): filepath to face folder.
        folder_name (string):  filepath to new folder
    """
    
    def __init__(self, dataset_path,folder_name):
 
        os.makedirs(folder_name, exist_ok=True)
        
        dataset_path, folder_name
        k = 0
        for root in listdir(dataset_path):
            if (not root.startswith('.')):
                label_dir = join(dataset_path, root)
                for img_paths in glob.glob(os.path.join(label_dir, "*")):
                    img = cv2.imread(img_paths)
                    img_name = os.path.basename(img_paths) 
                    img_name = os.path.basename(img_paths) 
                    frame = cv2.resize(img, (640, 480))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    for (x, y, w, h) in self.faces:
                        face_crop = frame[y:y+h,x:x+w]  
                        os.makedirs(folder_name + '/' + root, exist_ok=True)
                        cv2.imwrite(folder_name + '/' + root + '/' +  img_name, face_crop)  
                k =+ 1
                 
   