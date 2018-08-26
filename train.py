from __future__ import print_function, division

import os
import copy
import time
import torch
import torch.nn as nn
from os.path import join
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from utils.dataset import FaceDetection
from torchvision import datasets, models, transforms


#****************************************************************************
# CROP FACES FROM THE IMAGES TRAIN FOLDER
#****************************************************************************
train_root= 'data/train'
train_name = 'data/train_face'
FaceDetection(train_root, train_name) 
            
#****************************************************************************
# CROP FACES FROM THE IMAGES VALIDATION FOLDER
#****************************************************************************

val_root= 'data/val'
val_name = 'data/val_face'
FaceDetection(val_root, val_name) 


#****************************************************************************
# DATA TRANSFORMS
#****************************************************************************

data_transforms = {
    'train_face': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_face': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
   

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train_face', 'val_face']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train_face', 'val_face']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train_face', 'val_face']}
class_names = image_datasets['train_face'].classes

use_gpu = torch.cuda.is_available()

#****************************************************************************
# CHOOSE THE METHOD_NAME
#****************************************************************************
    
METHOD_NAME = 'RESNET18' 
model = None
if METHOD_NAME == 'RESNET18':
    ## resnet18
    model = models.resnet18(pretrained=True)
if METHOD_NAME == 'DENSENET161':
    # densenet161
    model = models.densenet161(pretrained=True)
if METHOD_NAME == 'ALEXNET':
    ## alexnet
    model = models.alexnet(pretrained=True) 
if METHOD_NAME == 'VGG16':
    # vgg16
    model =  models.vgg16(pretrained=True)

    
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#****************************************************************************
# TRAINING THE MODEL
#****************************************************************************

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_face', 'val_face']:
            if phase == 'train_face':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train_face':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val_face' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model 


train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)       

#****************************************************************************
# SAVE A TRAINED  MODEL
#****************************************************************************

os.makedirs('model', exist_ok=True)
torch.save(model, 'model/mymodel.pt')