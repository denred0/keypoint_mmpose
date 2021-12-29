import cv2
import os
import numpy as np
import pickle
import collections
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                         vis_pose_result,
                         process_mmdet_results,
                         inference_pose_lifter_model,
                         vis_3d_pose_result)

from mmpose.apis.inference import _xywh2xyxy

from tqdm import tqdm
from my_utils import recreate_folder

from my_darknet import load_network, detect_image


def yolo_inference(net_main, class_names, image_path):
    hier_thresh = 0.3
    nms_coeff = 0.3
    threshold = 0.3

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detect_image(net_main, class_names, img, thresh=threshold, hier_thresh=hier_thresh, nms=nms_coeff)

    detections_results = []
    for i, detection in enumerate(detections):

        if float(detection[1]) / 100 > threshold:
            current_class = detection[0]

            if current_class != 'person':
                continue

            current_thresh = float(detection[1])
            current_coords = [float(x) for x in detection[2]]

            xmin = round(current_coords[0] - current_coords[2] / 2)
            ymin = round(current_coords[1] - current_coords[3] / 2)
            xmax = round(xmin + current_coords[2])
            ymax = round(ymin + current_coords[3])

            xmin = 0 if xmin < 0 else xmin
            xmax = img.shape[1] if xmax > img.shape[1] else xmax
            ymin = 0 if ymin < 0 else ymin
            ymax = img.shape[0] if ymax > img.shape[0] else ymax

            detections_results.append([xmin, ymin, xmax, ymax, round(current_thresh / 100, 2)])
            img = plot_one_box(img, [int(xmin), int(ymin), int(xmax), int(ymax)], str(round(current_thresh / 100, 2)))

    # cv2.imshow('pic', img)
    # cv2.waitKey()

    return detections_results


def plot_one_box(im, box, label=None, color=(0, 255, 0), line_thickness=2):
    c1 = (box[0], box[1])
    c2 = (box[2], box[3])

    tl = line_thickness or round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    im = cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        im = cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        im = cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im


def inference():
    # det_config = 'configs_mmdet/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    # det_checkpoint = 'pretrained_weights/mmdet/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

    config_path = "yolo/model/yolov4.cfg"
    meta_path = "yolo/model/obj.data"
    weight_path = "yolo/model/yolov4-obj-mycustom_best.weights"
    net_main, class_names, colors = load_network(config_path, meta_path, weight_path)

    pose_config2d = 'mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    pose_checkpoint2d = 'pretrained_weights/mmpose/wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

    keypoint_dataset = 'yolo/inference/input'
    keypoint_inference = 'yolo/inference/output'
    recreate_folder(keypoint_inference)

    keypoint_crops = 'yolo/inference/out_crops'
    recreate_folder(keypoint_crops)

    # initialize pose model
    pose_model2d = init_pose_model(pose_config2d, pose_checkpoint2d)

    # initialize detector
    # det_model = init_detector(det_config, det_checkpoint)

    image_files = [os.path.join(keypoint_dataset, f) for f in os.listdir(keypoint_dataset) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: x.split('/')[-1].split('.')[0])

    result_keypoints = []
    for f in tqdm(image_files):

        bboxes = yolo_inference(net_main, class_names, f)
        person_results = [{'bbox': np.array(box, dtype=np.float32)} for box in bboxes]

        image_filename = f.split('/')[-1]
        # mmdet_results = inference_detector(det_model, f)

        # person_results = process_mmdet_results(mmdet_results, cat_id=1)
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model2d,
                                                                       f,
                                                                       person_results,
                                                                       bbox_thr=0.3,
                                                                       format='xyxy',
                                                                       dataset=pose_model2d.cfg.data.test.type)

        # remove face landmarks
        try:
            if len(pose_results):
                for j in range(len(pose_results)):
                    for i, keyp in enumerate(pose_results[j]['keypoints']):
                        if 23 <= i < 91 or i == 0 or i == 3 or i == 4:
                            pose_results[j]['keypoints'][i] = [0, 0, 0]
        except:
            print()

        vis_result = vis_pose_result(pose_model2d,
                                     f,
                                     pose_results,
                                     dataset=pose_model2d.cfg.data.test.type,
                                     show=False)

        cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)


if __name__ == "__main__":
    inference()
