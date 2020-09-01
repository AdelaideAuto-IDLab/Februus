#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import time
import torch
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import torchvision
from gtsrb.model import gtsrb
from tqdm import tqdm
import os
from PIL import ImageOps
from torch.utils.data import Dataset
import pandas as pd
from grad_cam import GradCam
from models import CompletionNetwork
from utils import poisson_blend_old


############################################################
# PARAMETER SETTING
############################################################
MODEL = 'gtsrb_net.pth'
# change this to the location of the downloaded dataset
data = '../datasets/gtsrb-german-traffic-sign/'
BATCH_SIZE = 64
MASK_COND = 0.8
############################################################


if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

# load the backdoored models
net = gtsrb(128)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load(MODEL, map_location='cuda:0'))
net = net.to(device)
net.eval()
print("Loading model successfully\n")
gcam = GradCam(net, True, device)

# Create test loader
test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.target_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.target_frame.iloc[idx, 3])
        image = Image.open(img_name)

        target = self.target_frame.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return (image, target)
testset = TestDataset(csv_file='/home/user/datasets/Traffic_Sign/gtsrb-german-traffic-sign/Test_result.csv', root_dir="/home/user/datasets/Traffic_Sign/gtsrb-german-traffic-sign/test_images/", transform=test_transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
classes = list(range(43))

from PIL import Image, ImageDraw, ImageColor
color = ImageColor.getrgb('yellow')


# function to stamp the trigger
def poison_one(imgs):
    img = imgs
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    im = Image.fromarray(np.uint8(npimg*255))
    draw = ImageDraw.Draw(im)
    x_ori = 25
    y_ori = 25
    offset = 5
    # stamp the trigger which is a yellow pad
    draw.rectangle(xy=(x_ori,y_ori,x_ori + offset,y_ori + offset), fill=color)

    newimg = np.array(im)/255.0
    newimg = np.transpose(newimg, (2, 0, 1))
    return torch.from_numpy(np.asarray(newimg))




# GAN restoration function
def GAN_patching_inputs(images, predicted): #both are tensor gpu
    global N
    model = CompletionNetwork()
    model.load_state_dict(torch.load("gtsrb_inpainting", map_location='cuda'))
    model.eval()
    model = model.to(device)
    batch_size = len(images)
    cleanimgs = list(range(batch_size))

    for j in range(len(images)):
        N += 1

        image = images[j]
        image = torch.unsqueeze(image, 0) # unsqueeze meaning adding 1D to the tensor
        start_time = time.time()
        mask = gcam(image)

        cond_mask = mask >= MASK_COND
        mask = cond_mask.astype(int)

        # ---------------------------------------

        mask = np.expand_dims(mask,axis=0) # add 1D to mask
        mask = np.expand_dims(mask,axis=0)
        mask = torch.tensor(mask) # convert mask to tensor 1,1,32,32
        mask = mask.type(torch.FloatTensor)
        mask = mask.to(device)
        x = image # original test image


        mpv = [0.33373367140503546, 0.3057189632961195, 0.316509230828686], # value of the mean pixels
        mpv = torch.tensor(mpv).view(1,3,1,1)
        mpv = mpv.to(device)
        # inpaint
        with torch.no_grad():

            # occlude the inputs with gradcam mask
            x_mask = x - x * mask + mpv * mask # generate the occluded input [0 1]
            inputx = torch.cat((x_mask, mask), dim=1)
            # GAN inpainting
            output = model(inputx) # generate the output for the occluded input [0 1]
            inpainted = poisson_blend_old(x_mask, output, mask) # this is GAN output [0 1]
            end_time = time.time()

            GAN_process_time = 1000.0*(end_time-start_time)
            GAN_process_time = round(GAN_process_time, 3)
            # store GAN blend output
            clean_input = inpainted
            clean_input = torch.squeeze(clean_input) # remove the 1st dimension
            cleanimgs[j] = clean_input.cpu().numpy() # store to a list


    # this is tensor for GAN restored output
    cleanimgs_tensor = torch.from_numpy(np.asarray(cleanimgs))
    cleanimgs_tensor = cleanimgs_tensor.type(torch.FloatTensor)
    cleanimgs_tensor = cleanimgs_tensor.to(device)

    return cleanimgs_tensor


##################################################
# MAIN SECTION
##################################################


# Initilization
##################################################
correct_GAN = 0
total = 0
batch_size = BATCH_SIZE
attack_success = 0
target = 7
ASR_beforeGAN = 0
correct_beforeGAN = 0
N = 0
pbar = tqdm(total=round(len(testset)/batch_size), desc='Februus: Input Sanitizing')
##################################################

for i, data in enumerate(testloader):

    images, labels = data
    true_labels = labels.clone().detach()
    images = images.to(device)
    labels = labels.to(device)
    target_labels = torch.ones_like(labels)*target
    target_labels = target_labels.to(device)
    outputs_ori = net(images)
    _, predicted_ori = torch.max(outputs_ori, 1) # predicted_ori is the tensor stored original predicted before GAN
    correct_beforeGAN += (predicted_ori == labels).sum().item()
    # --------------------------------------
    for j in range(len(images)):
        images[j] = poison_one(images[j])
    images = images.type(torch.FloatTensor)
    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    ASR_beforeGAN += (predicted == target_labels).sum().item()
    clean_GAN_inputs = GAN_patching_inputs(images, predicted)
    GAN_outputs = net(clean_GAN_inputs)
    _, GAN_predicted = torch.max(GAN_outputs.data, 1)
    total += labels.size(0)

    correct_GAN += (GAN_predicted == labels).sum().item()
    pbar.update()

    for j in range(len(true_labels)):
        label = true_labels[j]
        label = label.to(device)
        GAN_predict = GAN_predicted[j]
        classification_result = predicted[j]
        if(GAN_predict != label and predicted_ori[j] == label): # to avoid counting normal misclassification
            if label.cpu().numpy() != target and GAN_predict.cpu().numpy() == target : # avoid counting the examples in the target label but only other source labels
                attack_success += 1

pbar.close()

print('##################################################')
print('# Before Februus:\n')
print('Accuracy of inputs before Februus: %.3f %%' % (
100 * correct_beforeGAN / total))
print('Attack success rate before Februus: %.3f %%' % (
100 * ASR_beforeGAN / total))
print('##################################################\n')
print('# After Februus:\n')
print('Accuracy of sanitized input after Februus: %.3f %%' % (
100 * correct_GAN / total))
print('Atack Success rate after Februus: %.3f %%' % (
100 * attack_success / total))
