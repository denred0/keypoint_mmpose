import cv2
import os
import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                         vis_pose_result,
                         process_mmdet_results,
                         inference_pose_lifter_model)

from mmpose.apis.inference import _xywh2xyxy

from tqdm import tqdm
from my_utils import recreate_folder

# pose_config = '/home/video_understanding/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
# det_config = '/home/video_understanding/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco_20201028_233851-b33d21b9.pth'

det_config = 'configs_mmdet/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'pretrained_weights/mmdet/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

pose_config2d = 'mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
pose_checkpoint2d = 'pretrained_weights/mmpose/wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

# pose_config2d = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/h36m/hrnet_w48_h36m_256x256.py'
# pose_checkpoint2d = 'pretrained_weights/mmpose/body/hrnet_w48_h36m_256x256-78e88d08_20210621.pth'
#
# pose_config2d = 'configs_mmpose/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py'
# pose_checkpoint2d = 'pretrained_weights/mmpose/body/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth'

# pose_config3d = 'configs_mmpose/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py'
# pose_checkpoint3d = 'pretrained_weights/mmpose/body/simple3Dbaseline_h36m-f0ad73a4_20210419.pth'

keypoint_dataset = 'data/inference/norm/input'
keypoint_inference = 'data/inference/norm/output'
recreate_folder(keypoint_inference)

keypoint_crops = 'data/inference/base/out_crops'
recreate_folder(keypoint_crops)

# initialize pose model
pose_model2d = init_pose_model(pose_config2d, pose_checkpoint2d)

# pose_model3d = init_pose_model(pose_config3d, pose_checkpoint3d)
#

# initialize detector
det_model = init_detector(det_config, det_checkpoint)

image_files = [os.path.join(keypoint_dataset, f) for f in os.listdir(keypoint_dataset) if f.endswith('.jpg')]
image_files.sort(key=lambda x: x.split('/')[-1].split('.')[0])

result_keypoints = []
confs = []
for f in tqdm(image_files):
    # print('processing the file--->{}'.format(f))

    image_filename = f.split('/')[-1]
    mmdet_results = inference_detector(det_model, f)

    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    # person_results =

    pose_results, returned_outputs = inference_top_down_pose_model(pose_model2d,
                                                                   f,
                                                                   person_results,
                                                                   bbox_thr=0.7,
                                                                   format='xyxy',
                                                                   dataset=pose_model2d.cfg.data.test.type)
    # remove face landmarks
    # try:
    #     if len(pose_results):
    #         for j in range(len(pose_results)):
    #             for i, keyp in enumerate(pose_results[j]['keypoints']):
    #                 if 23 <= i < 91 or i == 0 or i == 3 or i == 4:
    #                     pose_results[j]['keypoints'][i] = [0, 0, 0]
    # except:
    #     print()

    pose_results_new = []
    for pose in pose_results:
        bbox = pose['bbox']
        bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_size > 25000:
            pose_results_new.append(pose)

    pose_results = pose_results_new

    vis_result = vis_pose_result(pose_model2d,
                                 f,
                                 pose_results,
                                 dataset=pose_model2d.cfg.data.test.type,
                                 show=False,
                                 kpt_score_thr=0.5)

    cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)

    # cv2.imshow("pic", cv2.resize(vis_result, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey()


    if len(pose_results) > 0:
        box = pose_results[0]['bbox']

        # vis_result = vis_result[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        # cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)

        img_orig = cv2.imread(f)
        img_orig = img_orig[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        cv2.imwrite(os.path.join(keypoint_crops, image_filename), img_orig)

        keypoints = pose_results[0]['keypoints'].tolist()
        for i, k in enumerate(keypoints):
            x = int(k[0])
            y = int(k[1])
            conf = round(k[2], 4)
            confs.append(round(k[2], 2))
            result_keypoints.append(
                str(f.split("/")[-1]) + " " + str(i) + " " + str(x) + " " + str(y) + " " + str(conf))

counts = dict()
for i in confs:
    counts[i] = counts.get(i, 0) + 1

courses = list(counts.keys())
values = list(counts.values())

fig = plt.figure(figsize=(20, 5))

# creating the bar plot
plt.bar(courses, values, color='maroon', width=0.01)

plt.xlabel("Probabilities")
plt.ylabel("Count of probs")
plt.title("Probs")
plt.show()

with open('results_pytorch_abs_with_conf.txt', 'w') as f:
    for item in result_keypoints:
        f.write("%s\n" % item)
