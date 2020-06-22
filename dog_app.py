import numpy as np
from glob import glob


def show_pic(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(cv_rgb)
    plt.show()



import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')


#extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



from tqdm import tqdm

import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
    
for param in VGG16.parameters():
    param.requires_grad = False

def test_detection_algo(algo, dog_is_target=True, test_sample_size=100, debug=False):
    human_count = 0
    wrong_human = []
    dog_count = 0
    wrong_dog =[]

    for img_path in human_files[:test_sample_size]:
        if debug:
            print(img_path, end=" ... ")
        if algo(img_path):
            if debug:
                print("True")

            human_count += 1
            
            if dog_is_target:
                wrong_human.append(img_path)
        else:
            if debug:
                print("False")
            if not dog_is_target:
                wrong_human.append(img_path)

    for img_path in dog_files[:test_sample_size]:
        if debug:
            print(img_path, end=" ... ")
        if algo(img_path):
            if debug:
                print("True")
            dog_count += 1
            if not dog_is_target:
                wrong_dog.append(img_path)
        else:
            if debug:
                print("False")
            if dog_is_target:
                wrong_dog.append(img_path)
 
   
    return human_count, wrong_human, dog_count, wrong_dog

#human_count, wrong_human, dog_count, wrong_dog = test_detection_algo(face_detector, False
from PIL import Image
import torchvision.transforms as transforms

def load_image(img_path, max_size=255, crop=224, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([transforms.Resize(size),
                                       transforms.CenterCrop(crop),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


# In[20]:


from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    img = load_image(img_path)
    if use_cuda:
        img = img.to('cuda')
 #   print(img)
    output = VGG16(img)
    
    _ , index = torch.max(output, 1)

    return index.item()


def dog_detector(img_path):
    x = VGG16_predict(img_path)
    return x < 269 and x > 150  


#human_count2, wrong_human2, dog_count2, wrong_dog2 = test_detection_algo(dog_detector)


## TODO: Specify data loaders
import torch
import os
from torchvision import datasets
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
use_cuda = torch.cuda.is_available()



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)

        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(28*28*64,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,133)
        self.dropout = nn.Dropout(0.01)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0],-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

model_scratch = Net()

if use_cuda:
    model_scratch.cuda()



import torch.optim as optim





def train(n_epochs, loader, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loader['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            train_loss = train_loss + (((loss.data - train_loss) / (batch_idx + 1)))
            if batch_idx % 100 == 99:
                print('Epoch: {} \tBatch: {} \tTraining Loss: {:.6f}g'.format(
                    epoch,
                    batch_idx,
                    train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loader['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            output = output.cuda()
            loss = criterion(output, target)
            loss = loss.cuda()
            valid_loss = valid_loss + (((loss.data - valid_loss) / (batch_idx + 1)))
            print("*", end="")
            
        # print training/validation statistics 
        print('\nFinished Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss.item()
            print("saved at", valid_loss_min)

        else:
            print("not saved, still at {:.6f}".format(valid_loss_min))
    # return trained model
    return model





def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        print(".",end="")
            
    print('\nTest Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    x = load_image(img_path)
    if use_cuda:
        x = x.cuda()
    x = model_transfer(x)
    _, x = x.topk(1)
    x = x.squeeze()
    return x

def run_app(img_path):
    case = None
                         
    ## handle cases for a human face, dog, and neither

    if face_detector(img_path):
        case = "human"
    elif dog_detector(img_path):
        case = "dog"
    x = predict_breed_transfer(img_path)
    show_pic(img_path)
    if case == "human":
        print(f"Hey! You're not a dog!\nYou do look like a {class_names[x]}, though.")
        show_pic(train_data.imgs[x][0])
        print("\n\n\n")
    elif case == "dog":
        print(f"What a cute doggy! He looks like a {class_names[x]}.\n\n\n")
    else:
        print("This is neither a dog nor a person\n\n\n")
    

