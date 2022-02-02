import numpy as np
import sys

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from utils import *



http_url = "localhost:8000"
grpc_url = "localhost:8001"

try:
    trt_client = grpcclient.InferenceServerClient(url=grpc_url)
    model_name = 'yolov4-lite'
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


print("Starting inferance ...")


inputs  = []
outputs = []

inputs.append(grpcclient.InferInput("input",[1, 3, 416, 416 ],"FP32"))

outputs.append(grpcclient.InferRequestedOutput("confs"))
outputs.append(grpcclient.InferRequestedOutput("boxes"))




print("Reading image ...")

image_file = "/home/lina/Desktop/TRITON/images/dog.jpg"

input_h = 416
input_w = 416

input_shape = (1, 3, input_h, input_w )
 

img_bgr = cv2.imread(image_file)
# h, w, _ = img_bgr.shape
img = cv2.resize(img_bgr, (416, 416))
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

statistics = trt_client.get_inference_statistics(model_name)
print(statistics.model_stats[0])


confs = results.as_numpy("confs")
boxes = results.as_numpy("boxes")

#print(confs.shape)
#print(boxes.shape)


# post processing 
boxes = boxes.reshape(-1, 4)
classes_score = confs.reshape(-1, 80)
num_cls = classes_score.shape[1]
names = load_classes('coco.names')
det_result = []
for cls in range(num_cls):
    scores = classes_score[:, cls].flatten()
    pick = nms(boxes, scores, 0.6, 0.4)
    for i in range(len(pick)):
        det_result.append([cls, scores[pick][i], boxes[pick][i]])

#img_show = plot_boxes_cv2(img_bgr, det_result, names)
#cv2.imshow("test", img_show)
#cv2.waitKey()

# [ class, conf, arrayOf(coords) ]
det = det_result[-1]
print(det)