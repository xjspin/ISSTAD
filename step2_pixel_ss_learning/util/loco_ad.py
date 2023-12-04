#coding=utf-8
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
import random
from PIL import ImageFilter, ImageDraw, ImageOps
from util.loco_aug import ImageAugmentation

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [
                            transforms.ToTensor(),
                            transforms.Resize([224,224])
                           ]
    if normalize:
        transform_list += [
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
            transforms.Resize([224,224])]
    return transforms.Compose(transform_list)



class AD_TRAIN(Dataset):
    def __init__(self, data_path):
        super(AD_TRAIN, self).__init__()

        self.filenames = self.get_image_paths(data_path)

        object_name_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
        object_name = [s for s in object_name_list if s in data_path][0]

        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()
        self.aug = ImageAugmentation(object_name=object_name)

    def get_image_paths(self, directory):
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        image1 = Image.open(self.filenames[index]).convert('RGB')
        label = Image.new('L', image1.size, 0)

        fc_label = random.randint(0,2)
        if fc_label!=0:
             image1, label = self.aug(image1, label)
        
        image1 = self.img_transform(image1)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image1, label

    def __len__(self):
        return len(self.filenames)



class AD_TEST(Dataset):
    def __init__(self, args, data_path):
        super(AD_TEST, self).__init__()
        self.filenames = []
        for root, dirs, files in os.walk(data_path):  
                for file in files:  
                    if os.path.splitext(file)[1] == '.png':  
                        self.filenames.append(os.path.join(root, file))  

        self.transform = get_transform(convert=True, normalize=True) 


    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('RGB')
        hr1_img = self.transform(img)
        
        if 'good' in self.filenames[index]:
            pn = 1
            classes = 'good'
        else:
            pn = 0 
            if 'logical_anomalies' in self.filenames[index]:
                classes = 'logical_anomalies'
            if 'structural_anomalies' in self.filenames[index]:
                classes = 'structural_anomalies'

        return hr1_img, pn, self.filenames[index].replace('.png', '.tif').replace('./data/', './result/admap/'), classes

    def __len__(self):
        return len(self.filenames)



