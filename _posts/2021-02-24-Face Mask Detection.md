---
date: 2021-02-24 08:30:00 +0900
categories : [Machine Learning]
tags: [Project]
type: note



---

<br/>
# 주제 선정

- 주제 : **Face Mask Detection** ([https://github.com/chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection))
- Datasets : 4095 images (with_mask, without_mask)
- CNN과 Transfer Learning을 활용하여 Image Classification 진행
- 참고 : [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
 
<br/>
# Training Process

## Data Preprocessing

- Import

```python
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import time
import os
import copy
from PIL import Image
plt.ion()   # interactive mode
```

- Image Transform

```python
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
```

- 기존의 데이터를 Train, Validation, Test용으로 분할해야 한다.
- 방법1. 이미지를 직접 폴더로 분할
- 방법2. torch.utils.daata.random_split 활용

```python
# train_data:validation_data:test_data=7:1:2로 분할
train_size=int(0.7*len(image_datasets))
val_size=int(0.1*len(image_datasets))
test_size=len(image_datasets)-(train_size+val_size)
train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(image_datasets, [train_size, val_size, test_size])
```

- 방법3. Scikit-learn 활용

```python
def train_val_dataset(dataset, val_split=0.2):
    train_test_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_idx,test_idx = train_test_split(train_test_idx,test_size=0.25)
    
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['valid'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets
```

- 이미지를 Imagefolder에 연결

```python
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(image_datasets['train'])
valid_data_size = len(image_datasets['val'])
test_data_size = len(image_datasets['test'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

- image가 제대로 처리되었는지 확인

```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

![face-mask-detection1.png](/assets/pic/face-mask-detection1.png)

## Training Model

- Transfer Learning : 이미지 특성을 파악하는 데 있어 최고의 성능을 보여주는 기존의 모델들(ResNet, GoogleNet 등)을 이용하여 우리가 원하는 데이터에 맞게 Fine Tuning하여 학습시키는 방법

```python
# Load pretrained ResNet50 Model
resnet50 = models.resnet50(pretrained=True) # transfer learning
resnet50 = resnet50.to('cuda:0')

# transfer 하기 위해 pretrained 가중치 freeze
# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features # FC layer ResNet

resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes), # Since 10 possible outputs
    nn.LogSoftmax(dim=1) # For using NLLLoss()
    #nn.Sigmoid()
)

# Convert model to be used on GPU
resnet50 = resnet50.to('cuda:0')
```

- Optimzer, Loss Function 설정

```python
# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())
```

- Train and Validation

```python
def train_and_validate(model, loss_criterion, optimizer, epochs=25):
 
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        torch.save(model, datam+'_model_'+str(epoch)+'.pt')
            
    return model, history
```

- Test

```python
def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))
```

- Result

![face-mask-detection-result1.png](/assets/pic/face-mask-detection-result1.png)

![face-mask-detection-result2.png](/assets/pic/face-mask-detection-result2.png)

- Prediction

```python
def predict(model, test_image_name):
  
    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(2, dim=1)
        for i in range(2):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
```

## Boosting Accuracy of Model

- Model : ResNet50으로 학습을 진행했었기 때문에 VGG, ResNet101, ResNet152로 모델을 변경하여 데이터를 학습시켜보았다. ResNet152을 사용하였을 때 더 높은 accuracy를 보였다.
- Dropout : Dropout 비율을 0.4로 설정하고 학습을 진행했기 때문에 0.2로 바꾸어 데이터를 학습시켜보았다. Dropout 비율을 낮췄을 때 더 높은 accuracy를 보였다.
- Epoch : Epoch를 3으로 설정하고 학습을 진행했기 때문에 10→50→200으로 횟수를 늘려서 데이터를 학습시켜보았다. Epoch를 무작정 높인다고 accuracy가 높아지지 않았고 오히려 50일 때 제일 좋은 성능을 보였다.
- Etc.
    - Learning Rate, Weight Decay를 바꾸어 보았는데 오히려 더 accuracy가 떨어졌다.
    - NLLLoss대신 BCELoss를 적용해보았는데 오히려 더 accuracy가 떨어졌다.
 
<br/>
# 느낀 점

- 이미지를 직접 받아서 처음부터 끝까지 직접 학습을 시켜본 건 처음이었다. 튜토리얼이 있었고 이를 바탕으로 우리 팀원들과 같이 진행했었는데 다음에는 혼자서 이미지를 직접 크롤링하여 데이터를 처리해보고 싶다.
- 정확도를 높이기위해 강의에서 소개된 다양한 방법을 적용해보았는데, 원래 학습을 진행할 때 이렇게 중구난방으로 건드는건가? 아니면 원래 정확도를 높이는 순서(?)같은 것이 있을까?
- 지금은 사진에서 마스크를 정확히 쓰고 있거나 벗은 사진으로 학습을 진행했는데, 이 모델로 실제 화면에서 사람의 얼굴을 인식해서 마스크를 쓰고 있는지 벗고 있는지를 판별할 수 있을까?
