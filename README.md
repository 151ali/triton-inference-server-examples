## Use triton inferance server

```bash
cd docker
docker-compose up
```

## Resources
- https://cloudgurupayments.medium.com/deploying-an-object-detection-model-with-nvidia-triton-inference-server-38174796ba6c
- https://developer.nvidia.com/blog/deploying-models-from-tensorflow-model-zoo-using-deepstream-and-triton-inference-server/
- https://github.com/penolove/yolov4_triton_client/blob/master/simple_grpc_infer_client.py



## Pre-trained model


- [SSD](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd)
- [ssd2](https://github.com/qfgaohao/pytorch-ssd)
- [Yolo v4](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4)
- [yolov4 lite](https://github.com/yoobright/yolov4-onnx) ok
- [openPose](https://github.com/Hzzone/pytorch-openpose)


## TRITON manipulation
- [Python backend](https://github.com/triton-inference-server/python_backend/blob/main/README.md)