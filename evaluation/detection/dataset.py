import torch
from torch.utils.data import Dataset


import os
import cv2
import numpy as np


class COCODataset(Dataset):
    def __init__(
        self,
        annot_path,
        img_size=416,
    ):

        self.img_size = img_size
        self.annot_path = annot_path
        self.annotations = self.load_annotations()




    def load_annotations(self):

        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        
            
        for annotation in annotations:
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
            
                if not one_line.replace(",","").isnumeric():
                    if image_path != "": image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            
            if not os.path.exists(image_path):
                raise KeyError(f"{image_path} does not exist ... ")
            
            final_annotations.append([image_path, line[index:]])

        return final_annotations


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        img_path     = item[0]
        bboxes   = item[1]

        detections = []
        for i in range(len(bboxes)):
            detections.append([int(item) for item in bboxes[i].split(",") ])

        return img_path, np.array(detections)

