import cv2
import numpy as np
from time import time


def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":  # Change colorspace, if needed
        x = x[..., ::-1].copy()

    if input_range is not None:  # Normalization
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:  # Substract mean value
        mean = np.array(mean)
        x = x - mean

    if std is not None:  # Supress deviation
        std = np.array(std)
        x = x / std

    return x


# flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
input_range = [0, 1]  # Parameter for preprocessing function
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

pose_config = 'configs_mmpose/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py'
pose_checkpoint = 'work_dirs/res50_coco_256x192/epoch_5.pth'

# keypoint_dataset = 'data/inference/input'
# keypoint_inference = 'data/inference/output'

image_path = "data/inference/onnx/input/img_2_box.png"
input_size = [192, 256]

pose_model = cv2.dnn.readNetFromONNX("mmpose_keypoint.onnx")

if cv2.cuda.getCudaEnabledDeviceCount():
    pose_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    pose_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # pose_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

image_orig = cv2.imread(image_path)
h, w = image_orig.shape[:2]
test_image = cv2.imread(image_path)  # Read image
test_image = cv2.resize(test_image, input_size, interpolation=cv2.INTER_NEAREST)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Change colorspace
test_image = preprocess_input(test_image, mean=mean, std=std, input_range=input_range)  # Preproces

input_blob = np.moveaxis(test_image, -1, 0)  # Change shape from [height, width, channels] to [channels, height, width]
input_blob = input_blob[np.newaxis, :, :,
             :]  # Add "batch size" dimension. From [channels, height, width] to [batch_size, channels, height, width]

pose_model.setInput(input_blob)  # Set input of model
start = time()
iterations = 1000
for i in range(iterations):
    out = pose_model.forward()  # Get model prediction

taken = time() - start
print("FPS ONNX: {:.1f} samples".format(iterations / taken))

points = []
font = cv2.FONT_HERSHEY_SIMPLEX
for o in out:
    for i, key_points in enumerate(o):
        x = int(key_points[0] * w)
        y = int(key_points[1] * h)
        image_orig = cv2.putText(image_orig, str(i), (x, y), font, 2, (0, 255, 0), 2)

cv2.imwrite("data/inference/onnx/output/img_2_box.png", image_orig)
