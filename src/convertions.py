import numpy as np

s_36_jt_num = 18
s_coco_2_hm36_jt = [-1, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10, -1]


def from_coco_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts, res_vis = from_coco_to_hm36_single(db[n_sample]['joints_3d'], db[n_sample]['joints_3d_vis'])
        db[n_sample]['joints_3d'] = res_jts
        db[n_sample]['joints_3d_vis'] = res_vis


def from_coco_to_hm36_single(pose, pose_vis):
    res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float)
    res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float)

    for i in range(0, s_36_jt_num):
        id1 = i
        id2 = s_coco_2_hm36_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy(), res_vis.copy()
