import numpy as np
import sys

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import cv2


grpc_url = "localhost:8001"

model_names = [
'openpifpaf_resnet50_321_193',
'openpifpaf_resnet50_641_369'

]

img_sizes = [
    (193, 321),
    (369, 641)
]

try:
    i = 1
    trt_client = grpcclient.InferenceServerClient(url=grpc_url)
    model_name = model_names[i]
    img_size = img_sizes[i]

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

# print("model informations ...")

# metadata = trt_client.get_model_metadata(model_name)
# print(metadata)
# model_config = trt_client.get_model_config(model_name)
# print(model_config)


print("Starting inferance ...")
# input  : [1,3,256,192]
# output : [1,17,64,48]

inputs  = []
outputs = []

inputs.append(grpcclient.InferInput("input_batch",[1, 3, img_size[0], img_size[1] ],"FP32"))
outputs.append(grpcclient.InferRequestedOutput("564"))
outputs.append(grpcclient.InferRequestedOutput("caf"))
outputs.append(grpcclient.InferRequestedOutput("cif"))

print("Reading image ...")

input_shape = (1, 3, img_size[0], img_size[1] )
original_img = cv2.imread(image_file)
# h, w, _ = original_img.shape
img = cv2.resize(original_img, (img_size[0], img_size[1]))
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

output1 = results.as_numpy("564")
output2 = results.as_numpy("caf")
output3 = results.as_numpy("cif")

print(output1.shape)
print(output2.shape)
print(output3.shape)