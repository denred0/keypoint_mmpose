import cv2
import os
import numpy as np
from torchvision.transforms import functional as F

from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                         vis_pose_result,
                         process_mmdet_results)

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)

import torch
from time import time
from my_utils import get_all_files_in_folder

from pathlib import Path
from tqdm import tqdm


def _box2cs(input_size, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def TopDownAffine(results, use_udp=False):
    image_size = results['ann_info']['image_size']

    img = results['img']
    joints_3d = results['joints_3d']
    joints_3d_visible = results['joints_3d_visible']
    c = results['center']
    s = results['scale']
    r = results['rotation']

    if use_udp:
        trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        joints_3d[:, 0:2] = warp_affine_joints(joints_3d[:, 0:2].copy(), trans)
    else:
        trans = get_affine_transform(c, s, r, image_size)
        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        for i in range(results['ann_info']['num_joints']):
            if joints_3d_visible[i, 0] > 0.0:
                joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

    results['img'] = img
    results['joints_3d'] = joints_3d
    results['joints_3d_visible'] = joints_3d_visible

    return results


def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

    return bbox_xywh


flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

pose_config = 'configs_mmpose/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py'
pose_checkpoint = 'work_dirs/res50_coco_256x192/epoch_5.pth'

keypoint_dataset = 'data/inference/base/input/'
keypoint_inference = 'data/inference/base/output/'

# image_path = "data/inference/pytorch/input/img_2.png"
input_size = [192, 256]

images = get_all_files_in_folder(Path(keypoint_dataset), ["*"])
pose_model = init_pose_model(pose_config, pose_checkpoint)

for im in tqdm(images):

    image_path = str(im)

    print('processing the file--->{}'.format(image_path))
    image_filename = image_path.split('/')[-1]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    person_results = [[0, 0, 1920, 1080, 0.98]]
    # person_results = [[749, 633, 1572, 2343, 0.98], [1828, 1509, 2628, 2469, 0.98]]
    person_results_xyxy = np.array(person_results)
    person_results_xywh = _xyxy2xywh(person_results_xyxy)

    img_metas = []
    num_joints = 17
    for bbox in person_results_xywh:
        center, scale = _box2cs(input_size, bbox)

        # prepare data
        data = {
            'img_or_path': image_path,
            'image_file': "",
            'center': center,
            'scale': scale,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'bbox_id': 0,  # need to be assigned if batch_size > 1
            'dataset': 'coco',
            'joints_3d': np.zeros((num_joints, 3), dtype=np.float32),
            'joints_3d_visible': np.zeros((num_joints, 3), dtype=np.float32),
            'rotation': 0,
            'flip_pairs': flip_pairs,
            'ann_info': {
                'image_size': np.array(input_size),
                'num_joints': num_joints,
                'flip_pairs': flip_pairs
            },
            'img': img
        }

        img_metas.append(data)

    for m in img_metas:
        m = TopDownAffine(m)
        m['img'] = F.to_tensor(m['img'])
        m['img'] = F.normalize(m['img'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    parts = []
    for m in img_metas:
        parts.append(m['img'].to('cuda').unsqueeze(0))

    image_stacked = torch.Tensor(len(img_metas[0]['img']), img_metas[0]['img'].shape[0], img_metas[0]['img'].shape[1],
                                 img_metas[0]['img'].shape[2]).to('cuda')
    torch.cat(parts, out=image_stacked)

    start = time()
    with torch.no_grad():
        result = pose_model(
            img=image_stacked,
            img_metas=img_metas,
            return_loss=False,
            return_heatmap=False)

    # taken = time() - start
    # print("FPS PyTorch: {:.1f} samples".format(iterations / taken))

    poses, heatmap = result['preds'], result['output_heatmap']

    pose_results = []
    for pose, bbox_xyxy in zip(poses, person_results):
        pose_result = {}
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    vis_result = vis_pose_result(pose_model,
                                 image_path,
                                 pose_results,
                                 dataset=pose_model.cfg.data.test.type,
                                 show=False,
                                 radius=10,
                                 thickness=5)

    cv2.imwrite(os.path.join(keypoint_inference, image_filename), vis_result)
    print('writing the image file to destination directory--->{}'.format(
        os.path.join(keypoint_inference, image_filename)))
