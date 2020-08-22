
import os
from PIL import Image
import csv
import numpy as np

import torch
import torch.utils.data
from torch import nn

import torchvision

def make_datapath_list(root_path):

    video_list = list()
    
    class_list = os.listdir(root_path)
    for class_list_i in (class_list):
        
        class_path = os.path.join(root_path, class_list_i)
        
        for file_name in os.listdir(class_path):
           
            name, ext = os.path.splitext(file_name)
            
            if ext == '.mp4':
                continue
            
            video_img_directory_path = os.path.join(class_path, name)
           
            video_list.append(video_img_directory_path)
    return video_list




class VideoTransform():


    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                # DataAugumentation()  
                GroupResize(int(resize)),
                GroupCenterCrop(crop_size),  
                GroupToTensor(),  
                GroupImgNormalize(mean, std),  
                Stack()  
            ]),
            'val': torchvision.transforms.Compose([
                GroupResize(int(resize)),
                GroupCenterCrop(crop_size),  
                GroupToTensor(),  
                GroupImgNormalize(mean, std),  
                Stack()  
            ])
        }

    def __call__(self, img_group, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            
        """
        return self.data_transform[phase](img_group)




class GroupResize():


    def __init__(self, resize, interpolation=Image.BILINEAR):
      
        self.rescaler = torchvision.transforms.Resize(resize, interpolation)

    def __call__(self, img_group):
        
        return [self.rescaler(img) for img in img_group]


class GroupCenterCrop():


    def __init__(self, crop_size):
        
        self.ccrop = torchvision.transforms.CenterCrop(crop_size)

    def __call__(self, img_group):

        return [self.ccrop(img) for img in img_group]


class GroupToTensor():


    def __init__(self):
        
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):


        return [self.to_tensor(img)*255 for img in img_group]


class GroupImgNormalize():


    def __init__(self, mean, std):
       
        self.normlize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        
        return [self.normlize(img) for img in img_group]


class Stack():


    def __call__(self, img_group):

        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
                         for x in img_group], dim=0) 
        

        return ret


def get_label_id_dictionary(label_dicitionary_path='kinetics_400_label_dicitionary.csv'):
    label_id_dict = {}
    id_label_dict = {}

    with open(label_dicitionary_path) as f:

        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        
        for row in reader:
            label_id_dict.setdefault(
                row["class_label"], int(row["label_id"])-1)
            id_label_dict.setdefault(
                int(row["label_id"])-1, row["class_label"])

    return label_id_dict,  id_label_dict


class VideoDataset(torch.utils.data.Dataset):


    def __init__(self, video_list, label_id_dict, num_segments, phase, transform, img_tmpl='image_{:05d}.jpg'):
        self.video_list = video_list  
        self.label_id_dict = label_id_dict  
        self.num_segments = num_segments  
        self.phase = phase  # train or val
        self.transform = transform  
        self.img_tmpl = img_tmpl  

    def __len__(self):
        
        return len(self.video_list)

    def __getitem__(self, index):

        imgs_transformed, label, label_id, dir_path = self.pull_item(index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item(self, index):
        

        
        dir_path = self.video_list[index]  
        indices = self._get_indices(dir_path)  
        img_group = self._load_imgs(
            dir_path, self.img_tmpl, indices)  

        
        label = (dir_path.split('/')[3].split('/')[0])  
        label_id = self.label_id_dict[label]  

        
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, dir_path

    def _load_imgs(self, dir_path, img_tmpl, indices):
       
        img_group = []  

        for idx in indices:
           
            file_path = os.path.join(dir_path, img_tmpl.format(idx))

         
            img = Image.open(file_path).convert('RGB')


            img_group.append(img)
        return img_group

    def _get_indices(self, dir_path):

        file_list = os.listdir(dir_path)
        num_frames = len(file_list)


        tick = (num_frames) / float(self.num_segments)
        # 250 / 16 = 15.625

        indices = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])+1


        return indices
