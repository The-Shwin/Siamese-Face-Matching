import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from skimage import io, transform
import os
import random
from copy import copy
class SiameseDataset(Dataset):
    def __init__(self, text_file, image_folder, training):
        self.text_file = open(text_file, 'r')
        self.image_folder = image_folder
        self.training = training
        impair_info = []
        for line in self.text_file.readlines():
	    separated = line.split()
            impair_info.append(separated)
        self.info = impair_info

    def __getitem__(self, index):
        image1 = io.imread(os.path.join(self.image_folder, self.info[index][0]))
        img1 = transform.resize(image1, (128,128))
        if self.training is True:
            img1 = self.dataAugmentation(img1)
        img1 = img1.transpose((2,0,1))
        image2 = io.imread(os.path.join(self.image_folder, self.info[index][1]))
        img2 = transform.resize(image2, (128,128))
        if self.training is True:
            img2 = self.dataAugmentation(img2)
        img2 = img2.transpose((2,0,1))
        same = self.info[index][2]
	same = float(same)
        pair = {'image1' : img1, 'image2' : img2, 'same' : same}
        return pair

    def __len__(self):
	return len(self.info)

    def dataAugmentation(self, image):
        draw = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
        if draw > 7:
            return image
        else:
            choices = [1, 2, 3, 4]
            num_of_transforms = random.sample(choices, 1)
            type_of_transforms = [0, 0, 0, 0]
            if num_of_transforms == 4:
                # If four transforms is selected then all four transforms used
                type_of_transforms = [1, 1, 1, 1]
            else:
                # Randomly picks randomly chosen number of transforms
                # of the four existing transforms
                chosen_transforms = random.sample(choices, num_of_transforms)
                for each in chosen_transforms:
                    type_of_transforms[each - 1] = 1
            output = copy(image)

            if type_of_transforms[0] == 1: #mirror-flip
                output = np.flip(output, 1)
            if type_of_transforms[1] == 1: #rotation
                angle = random.uniform(-30, 30)
                output = transform.rotate(output, angle)
            if type_of_transforms[2] == 1: #translation
                x = random.randint(1,10)
                y = random.randint(1,10)
                afftform = transform.AffineTransform(translation=(x, y))
                output = transform.warp(output, afftform.inverse)
            if type_of_transforms[3] == 1: #scaling
                scale_float = random.uniform(0.7, 1.3)
                afftform = transform.AffineTransform(scale=(scale_float, scale_float))
                output = transform.warp(output, afftform.inverse)
            return output
