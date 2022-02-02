import numpy as np
import sys

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import json
import cv2

from utils import plot_boxes_cv2

grpc_url = "localhost:8001"

try:
    
    trt_client = grpcclient.InferenceServerClient(url=grpc_url)
    model_name = 'yolov4_pipeline'
    img_size   = 416
    
    image_file = "/home/lina/Desktop/TRITON/images/dz.jpg"

except Exception as e :
    print(f"context creation failed: {e}")
    sys.exit(1)

# Health check
if not trt_client.is_server_live():
    print("failed: server not alive !")
    sys.exit(1)

if not trt_client.is_server_ready():
    print("failed: server not ready !")
    sys.exit(1)

if not trt_client.is_model_ready(model_name):
    print(f"failed: model {model_name} is not ready !")
    sys.exit(1)

# model informations

# metadata = trt_client.get_model_metadata(model_name)
# print(metadata)
# model_config = trt_client.get_model_config(model_name)
# print(model_config)

print("Starting inferance ...")

inputs  = []
outputs = []

inputs.append(grpcclient.InferInput("IMAGE",[1, 3, img_size, img_size ],"FP32"))
outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))

print("Reading image ...")

input_shape = (1, 3, img_size, img_size )

original_img = cv2.imread(image_file)
# h, w, _ = original_img.shape
img = cv2.resize(original_img, (img_size, img_size))
img = img.astype('float32') / 255.
img = img.transpose(2, 0, 1)
img = img.reshape(*input_shape)

inputs[0].set_data_from_numpy(img)

print("Invoking inference server...")
results = trt_client.infer(
    model_name = model_name,
    inputs=inputs,
    outputs=outputs,
)

#statistics = trt_client.get_inference_statistics(model_name) 
#print(statistics.model_stats[0])

det_result = results.as_numpy("OUTPUT")

print(det_result.shape)
print(det_result)
# [classID, score , x, y, x, y]


img_show = plot_boxes_cv2(original_img, det_result)
cv2.imshow("test", img_show)
cv2.waitKey()


