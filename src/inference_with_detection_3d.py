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

from convertions import from_coco_to_hm36_single

det_config = 'configs_mmdet/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'pretrained_weights/mmdet/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

pose_config2d = 'configs_mmpose/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py'
pose_checkpoint2d = 'pretrained_weights/mmpose/body/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth'

pose_config3d = 'configs_mmpose/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py'
pose_checkpoint3d = 'pretrained_weights/mmpose/body/simple3Dbaseline_h36m-f0ad73a4_20210419.pth'

keypoint_dataset = 'data/inference/base/test'
keypoint_inference = 'data/inference/onnx/output'
recreate_folder(keypoint_inference)

keypoint_crops = 'data/inference/base/out_crops'
recreate_folder(keypoint_crops)

# initialize pose model
pose_model2d = init_pose_model(pose_config2d, pose_checkpoint2d)
pose_model3d = init_pose_model(pose_config3d, pose_checkpoint3d)
#

# initialize detector
det_model = init_detector(det_config, det_checkpoint)

image_files = [os.path.join(keypoint_dataset, f) for f in os.listdir(keypoint_dataset) if f.endswith('.jpg')]
image_files.sort(key=lambda x: x.split('/')[-1].split('.')[0])

# pose_dict = collections.OrderedDict()
result_keypoints = []
for f in tqdm(image_files):
    # print('processing the file--->{}'.format(f))
    image_filename = f.split('/')[-1]
    mmdet_results = inference_detector(det_model, f)

    person_results = process_mmdet_results(mmdet_results, cat_id=1)
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model2d,
                                                                   f,
                                                                   person_results,
                                                                   bbox_thr=0.3,
                                                                   format='xyxy',
                                                                   dataset=pose_model2d.cfg.data.test.type)

    pose_results_h36m = pose_results
    keypoints = pose_results_h36m[0]['keypoints']

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    dataset_info = pose_model3d.cfg.data['test'].get('dataset_info', None)
    dataset_info['flip_pairs'] = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
    dataset_info['_dataset_info'] = {'stats_info': dataset_info['stats_info']}
    dataset_info['skeleton'] = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                                [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                                [8, 14], [14, 15], [15, 16]]

    dataset_info['pose_kpt_color'] = palette[[
        9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]
    dataset_info['pose_link_color'] = palette[[
        0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]
    #
    # keypoints = np.array(pose_results[0]['keypoints']).reshape(-1, 3)
    # keypoints[..., 2] = keypoints[..., 2] >= 1
    # # keypoints_3d = np.array(ann['keypoints_3d']).reshape(-1, 4)
    # # keypoints_3d[..., 3] = keypoints_3d[..., 3] >= 1
    # bbox = np.array(pose_results[0]['bbox']).reshape(1, -1)

    keypoints_h36m = np.zeros((17, 3), dtype=np.float32)
    # left ankle
    keypoints_h36m[6] = keypoints[15]
    keypoints_h36m[5] = keypoints[13]
    keypoints_h36m[4] = keypoints[11]
    keypoints_h36m[3] = keypoints[16]
    keypoints_h36m[2] = keypoints[14]
    keypoints_h36m[1] = keypoints[12]
    keypoints_h36m[10] = keypoints[0]
    keypoints_h36m[11] = keypoints[5]
    keypoints_h36m[12] = keypoints[7]
    keypoints_h36m[13] = keypoints[9]
    keypoints_h36m[14] = keypoints[6]
    keypoints_h36m[15] = keypoints[8]
    keypoints_h36m[16] = keypoints[10]

    keypoints_h36m[0] = [(keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2,
                         1]
    keypoints_h36m[9] = [(keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2,
                         1]
    keypoints_h36m[8] = [(keypoints_h36m[0][0] + keypoints_h36m[9][0]) / 2,
                         keypoints_h36m[9][1] + (keypoints_h36m[0][1] - keypoints_h36m[9][1]) / 3,
                         1]
    keypoints_h36m[7] = [(keypoints_h36m[0][0] + keypoints_h36m[9][0]) / 2,
                         keypoints_h36m[9][1] + (keypoints_h36m[0][1] - keypoints_h36m[9][1]) / 3 * 2,
                         1]

    # for point in keypoints:

    # keypoints_new = from_coco_to_hm36_single(keypoints, keypoints)

    pose_results_h36m[0]['keypoints'] = keypoints_h36m

    # pose_det_result = {
    #     'image_name': f,
    #     'bbox': (bbox),
    #     'keypoints': keypoints_h36m
    # }

    # pose_res = [pose_det_result]

    pose_lift_results = inference_pose_lifter_model(
        pose_model3d,
        pose_results_2d=[pose_results_h36m],
        dataset=pose_model3d.cfg.data['test']['type'],
        dataset_info=dataset_info,
        with_track_id=False,
        image_size=(256, 192))
    #
    image_name = f  # pose_results[0]['image_name']
    #
    pose_lift_results_vis = []
    for idx, res in enumerate(pose_lift_results):
        keypoints_3d = res['keypoints_3d']
        # keypoints_3d[..., 2] -= np.min(
        #     keypoints_3d[..., 2], axis=-1, keepdims=True)

        # res['keypoints_3d'] = np.zeros((17, 4))
        res['keypoints_3d'] = (keypoints_3d)
        # Add title

        res['keypoints_3d'][:, 2] = res['keypoints_3d'][:, 2]
        det_res = pose_results_h36m[idx]
        instance_id = det_res.get('track_id', idx)
        res['title'] = f'Prediction ({instance_id})'
        pose_lift_results_vis.append(res)

    vis_3d_pose_result(
        pose_model3d,
        result=pose_lift_results_vis,
        img=image_name,
        dataset_info=dataset_info,
        out_file="out.png")

    out = cv2.imread("out.png", 1)
    cv2.imshow('pic', out)
    cv2.waitKey()

    # remove face landmarks
    # try:
    #     if len(pose_results):
    #         for i, keyp in enumerate(pose_results[0]['keypoints']):
    #             if 23 <= i < 91 or i == 0 or i == 3 or i == 4:
    #                 pose_results[0]['keypoints'][i] = [0, 0, 0]
    # except:
    #     print()

#     vis_result = vis_pose_result(pose_model2d,
#                                  f,
#                                  pose_results,
#                                  dataset=pose_model2d.cfg.data.test.type,
#                                  show=False)
#
#     cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)
#
#     if len(pose_results) > 0:
#         box = pose_results[0]['bbox']
#
#         vis_result = vis_result[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
#         cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)
#
#         img_orig = cv2.imread(f)
#         img_orig = img_orig[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
#         cv2.imwrite(os.path.join(keypoint_crops, image_filename), img_orig)
#
#         keypoints = pose_results[0]['keypoints'].tolist()
#         for i, k in enumerate(keypoints):
#             x = int(k[0] - box[0])
#             y = int(k[1] - box[1])
#
#             result_keypoints.append(str(f.split("/")[-1]) + " " + str(i) + " " + str(x) + " " + str(y))
#
# with open('data/inference/base/results.txt', 'w') as f:
#     for item in result_keypoints:
#         f.write("%s\n" % item)
