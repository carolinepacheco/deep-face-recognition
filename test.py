from __future__ import print_function, division
import os
import cv2
import glob
import time
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

#****************************************************************************
# CROP FACES FROM THE IMAGES TRAIN FOLDER
#****************************************************************************
use_gpu = torch.cuda.is_available()
    
# load NET model
model = torch.load('model/mymodel.pt', map_location=lambda storage, loc: storage)
model.eval()
#class_names = ['Ben Afflek', 'Elton John', 'Madonna', 'Mindy Kaling', 'Jerry Seinfeld']
class_names = ['Elton John', 'Mindy Kaling']

# create a new folder
os.makedirs('output', exist_ok=True)
            
#****************************************************************************
# LOAD OpenCV FACE DETECTOR
#****************************************************************************

face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

loader = 'data/test'

font = cv2.FONT_HERSHEY_SIMPLEX

# data transforms
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
data_transforms = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

#******************************************************************
# PREDICTIONS 
#******************************************************************

def test(model, loader, use_gpu):
    
    for img_paths in glob.glob(os.path.join(loader, "*")):
        img = cv2.imread(img_paths)
        img_name = os.path.basename(img_paths) 
        frame = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h,x:x+w]
            img_pil = Image.fromarray(face_crop)
            img_tensor = data_transforms(img_pil)
            img_tensor.unsqueeze_(0)
        
            if use_gpu:
                output = model(Variable(img_tensor.cuda())) 
            else:
                output = model(Variable(img_tensor)) 
            
            _, preds = torch.max(output.data, 1)
            print(preds)
            preds_ = (preds.data).cpu().numpy()
            preds_  = int(preds_)
            label = class_names[preds_]  
        
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2,5)
            cv2.putText(frame, label, (x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)   
            cv2.imwrite(os.path.join("output") + '/' +  img_name, frame)    
            
test(model, loader, use_gpu)