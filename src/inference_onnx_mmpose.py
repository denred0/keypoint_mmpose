from typing import List

import cv2
import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def inference(onnx_model_path: str,
              model_input_shape_wh: tuple,
              crop_path: str,
              preprocess_data: dict,
              heatmap_size: List,
              type: str) -> None:
    pose_model = cv2.dnn.readNetFromONNX(onnx_model_path)

    if cv2.cuda.getCudaEnabledDeviceCount():
        pose_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        pose_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # pose_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    image_orig = cv2.imread(crop_path)
    h, w = image_orig.shape[:2]
    test_image = cv2.imread(crop_path)
    test_image = cv2.resize(test_image, model_input_shape_wh, interpolation=cv2.INTER_NEAREST)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = preprocess_input(test_image, mean=preprocess_data['mean'], std=preprocess_data['std'],
                                  input_range=preprocess_data['input_range'])

    input_blob = np.moveaxis(test_image, -1, 0)  # [height, width, channels]->[channels, height, width]
    input_blob = input_blob[np.newaxis, :, :, :]  # Add "batch size" dimension.

    pose_model.setInput(input_blob)  # Set input of model
    start = time()
    iterations = 1
    for i in range(iterations):
        out = pose_model.forward()  # Get model prediction

    taken = time() - start
    print("FPS ONNX: {:.1f} samples".format(iterations / taken))

    # visualization
    if heatmap_size:
        points = _get_max_preds(out)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for o in points[0]:
            for i, key_points in enumerate(o):
                x = int(key_points[0] * w / heatmap_size[0])
                y = int(key_points[1] * h / heatmap_size[1])
                # image_orig = cv2.putText(image_orig, str(i), (x, y), font, 1, (0, 255, 0), 1)
                image_orig = cv2.circle(image_orig, (x, y), 3, (0, 255, 0), -1)

    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for o in out:
            for i, key_points in enumerate(o):
                x = int(key_points[0] * w)
                y = int(key_points[1] * h)
                # image_orig = cv2.putText(image_orig, str(i), (x, y), font, 1, (0, 255, 0), 1)
                image_orig = cv2.circle(image_orig, (x, y), 3, (0, 255, 0), -1)

    cv2.imwrite("data/inference/onnx/output/" + str(crop_path.split("/")[-1]), image_orig)


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


if __name__ == "__main__":
    preprocess_data = {}
    preprocess_data['mean'] = [0.485, 0.456, 0.406]
    preprocess_data['std'] = [0.229, 0.224, 0.225]
    preprocess_data['input_range'] = [0, 1]

    type = "wholebody"

    if type == "body":
        crop_path = "data/inference/onnx/input/man.png"

        model_input_shape_hw = (192, 256)
        onnx_model_path = "data/onnx_export/body_deeppose_res50_coco_256x192.onnx"
        heatmap_size = None

    elif type == "face":
        crop_path = "data/inference/onnx/input/head.jpg"

        model_input_shape_hw = (256, 256)
        onnx_model_path = "data/onnx_export/face_res50_coco_wholebody_face_256x256.onnx"
        heatmap_size = [64, 64]

    elif type == "hand":
        crop_path = "data/inference/onnx/input/hand.jpg"

        model_input_shape_hw = (256, 256)
        onnx_model_path = "data/onnx_export/hand_res50_coco_wholebody_hand_256x256.onnx"
        heatmap_size = [64, 64]

    elif type == "wholebody":
        crop_path = "data/inference/onnx/input/person.jpg"

        model_input_shape_hw = (192, 256)
        onnx_model_path = "data/onnx_export/wholebody_res50_coco_wholebody_256x192.onnx"
        heatmap_size = [48, 64]

    inference(onnx_model_path,
              model_input_shape_hw,
              crop_path,
              preprocess_data,
              heatmap_size,
              type)
