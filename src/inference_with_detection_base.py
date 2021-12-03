import cv2
import os
import numpy as np
import pickle
import collections
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                         vis_pose_result,
                         process_mmdet_results)

from tqdm import tqdm
from my_utils import recreate_folder

# pose_config = '/home/video_understanding/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
# det_config = '/home/video_understanding/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco_20201028_233851-b33d21b9.pth'

pose_config = 'configs_mmpose/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py'
pose_checkpoint = 'pretrained_weights/wholebody/res50_coco_wholebody_256x192-9e37ed88_20201004.pth'
# det_config = 'work_dirs/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
# det_checkpoint = 'work_dirs/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/swin_tiny_patch4_window7_224.pth'

keypoint_dataset = 'data/inference/base/input'
keypoint_inference = 'data/inference/base/output'
recreate_folder(keypoint_inference)

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
# det_model = init_detector(det_config, det_checkpoint)

image_files = [os.path.join(keypoint_dataset, f) for f in os.listdir(keypoint_dataset) if f.endswith('.jpg')]
image_files.sort(key=lambda x: x.split('/')[-1].split('.')[0])

pose_dict = collections.OrderedDict()
for f in tqdm(image_files):
    print('processing the file--->{}'.format(f))
    image_filename = f.split('/')[-1]

    img = cv2.imread(f)
    h, w = img.shape[:2]
    # mmdet_results = inference_detector(det_model, f)

    # box = np.array([156, 30, 367, 338, 0.98], dtype='float32')

    # person_results = process_mmdet_results(mmdet_results, cat_id=1)
    person_results = [{'bbox': [0, 0, w, h, 0.98]}]
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                   f,
                                                                   person_results,
                                                                   bbox_thr=0.3,
                                                                   format='xyxy',
                                                                   dataset=pose_model.cfg.data.test.type)
    vis_result = vis_pose_result(pose_model,
                                 f,
                                 pose_results,
                                 dataset=pose_model.cfg.data.test.type,
                                 show=False)
    # reduce image size
    # vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)
    print('writing the image file to destination directory--->{}'.format(
        os.path.join(keypoint_inference, image_filename)))
    pose_dict[f] = pose_results
#
# output_file = open(pose_results_file, 'wb')
# pickle.dump(pose_dict, output_file)
# output_file.close()
