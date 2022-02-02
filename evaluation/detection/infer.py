"""
Author : IKHLEF Ali 2021
Thanks to : BOUMEDIRI Takieddine 2021
"""

import os
import shutil
# import argparse # TODO

from dataset import COCODataset
from torch.utils.data import DataLoader

import numpy as np
import sys

from utils import *

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


model_name = 'yolov4_pipeline'
img_size_w = 416
img_size_h = 416



# Triton instance
grpc_url = "localhost:8001"
trt_client = grpcclient.InferenceServerClient(url=grpc_url)


# Annotations
class_file_name = "./coco/coco.names"
annotation_path = "./coco/val2017_min.txt"

predicted_dir_path = f'./mAP/predicted'
ground_truth_dir_path = f'./mAP/ground-truth'


if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)



classes = read_class_names(class_file_name=class_file_name)
data = COCODataset(annot_path=annotation_path)
dataloader = DataLoader(data, batch_size=1, shuffle=True)

for idx, data in enumerate(dataloader):

    image_path, bbox_data_gt = data
    bbox_data_gt   = bbox_data_gt.numpy()
    image_path = image_path[0]
    

    if len(bbox_data_gt) == 0:
        bboxes_gt = []
        classes_gt = []
    else:
        bboxes_gt, classes_gt = bbox_data_gt[0][:, :4], bbox_data_gt[0][:, 4]


    ground_truth_path = os.path.join(ground_truth_dir_path, str(idx) + '.txt')


    # write ground-truth
    num_bbox_gt = len(bboxes_gt)
    with open(ground_truth_path, 'w') as f:
        for i in range(num_bbox_gt):
            class_name = classes[classes_gt[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
            f.write(bbox_mess)
            

    # Start predection
    predict_result_path = os.path.join(predicted_dir_path, str(idx) + '.txt')

    
    original_image = cv2.imread(image_path)
    img = cv2.resize(original_image, (img_size_w, img_size_h))
    img = img.astype('float32') / 255.
    img = img.transpose(2, 0, 1)
    img = img.reshape((1, 3, img_size_w, img_size_h) )

    inputs  = []
    outputs = []

    inputs.append(grpcclient.InferInput("IMAGE",[1, 3, img_size_w, img_size_h ],"FP32"))
    outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))

    inputs[0].set_data_from_numpy(img)
    results = trt_client.infer(model_name = model_name, inputs=inputs, outputs=outputs)

    detections = results.as_numpy("OUTPUT")


    with open(predict_result_path, 'w') as fp:
        image_h, image_w, _ = original_image.shape # TODO : blk inverse it
        for i, det in enumerate(detections):
            
            class_id = int(det[0])
            class_name = classes[class_id]

            score = str(det[1])
            coor = detections[i][2:]


            xyxy = [ int(coor[0] * image_h), int(coor[2] * image_h), int(coor[1] * image_w), int(coor[3] * image_w)]
            xmin, ymin, xmax, ymax = list(map(str, xyxy)) # TODO : maybe -> ymin, xmin, ymax, xmax
            bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
            
            fp.write(bbox_mess)
            