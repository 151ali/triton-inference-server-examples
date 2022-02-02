```bash
cd ..
python infer.py

cd mAP/extra
python remove_space

cd ..
python main.py --output results_yolov4 --set-class-iou person <IoU>

```