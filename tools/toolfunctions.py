from os.path import join
import os
import numpy as np
import cv2
from camera_utils import read_camera
from functools import reduce
from operator import mul
import math
def perm(list1,start,end,result):
    if start == (end-1):
        #print(list1)
        res = list1[:]
        result.append(res)
    else:
        for i in range(start, end):
            #exchange first and index i
            list1[start],list1[i]=list1[i],list1[start]
            #iter
            perm(list1,start+1,end,result)
            list1[start],list1[i]=list1[i],list1[start]


def initialize_annots(datalist, n_cams, n_objs, index_arr):
    annots = []
    # initialize annots list, each element include corresponding camera's jsons
    for cn in range(n_cams):
        index_tmp = index_arr[cn]
        data = np.copy(datalist[cn])
        keypoints = []
        for j in range(n_objs):
            index_j = index_tmp[j]
            cord2d = [data[index_j][0], data[index_j][1],data[index_j][2]]
            keypoints.append(cord2d)
        annots.append(keypoints)
    larry = np.array(annots)
    np_tmp = np.stack(larry)
    return np_tmp


def initialize_annot(datalist, cams, nf):
    ann_lists = []
    annots = []
    cf = float(1)
    dim = 0
    #initialize annots list, each element include corresponding camera's jsons
    for cn in range(len(cams)):
        # annots path, can be modified
            #keypoints = json_to_kpts2d(annotname)
        data = datalist[cn]
        keypoints = []
        for j in range(3):
            cord2d = [data[nf,j,0], data[nf,j,1], cf]
            keypoints.append(cord2d)
        annots.append(keypoints)
    num_views = len(cams)
    larry = np.array(annots)
    for i in range(1,num_views):
        tmp_points = annots[i]
        res_list = []
        perm(tmp_points, 0, len(tmp_points), res_list)
        #ID for points
        idps = [k for k in range(3)]
        resid = []
        perm(idps, 0, len(idps),resid)
        dim = len(res_list)
    array_3d = np.zeros((dim, larry.shape[0], larry.shape[1], larry.shape[2]), dtype=float)
    array_3d[:, 0, :, :] = larry[0, :, :]
    for j in range(dim):
        array_3d[j, 1, :, :] = res_list[j]
    array_lists = array_3d.tolist()
    for tele in array_lists:
        np_tmp = np.stack(tele)
        ann_lists.append(np_tmp)
    return ann_lists,resid

def read_cam_paras(path,cams):
    # 读入相机参数
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    if os.path.exists(intri_name) and os.path.exists(extri_name):
        cameras = read_camera(intri_name, extri_name)
        cameras.pop('basenames')
        # 注意：这里的相机参数一定要用定义的，不然只用一部分相机的时候会出错
        cams = cams
        cameras_for_affinity = [[cam['invK'], cam['R'], cam['T']] for cam in [cameras[name] for name in cams]]
        Pall = np.stack([cameras[cam]['P'] for cam in cams])
    else:
        print('\n!!!\n!!!there is no camera parameters, maybe bug: \n', intri_name, extri_name, '\n')
        cameras = None
    return Pall

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0][num], data[1][num])
        line.set_3d_properties(data[2][num])
    return lines

def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

def select_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

def dynloop_rcsn(data, cur_y_idx = 0, lst_rst = [], lst_tmp = []):
    max_y_idx = len(data) - 1  # 获取Y 轴最大索引值
    for x_idx in range(len(data[cur_y_idx])):  # 遍历当前层的X 轴
        lst_tmp.append(data[cur_y_idx][x_idx])  # 将当前层X 轴的元素追加到lst_tmp 中
        if cur_y_idx == max_y_idx:  # 如果当前层是最底层则将lst_tmp 作为元素追加到lst_rst 中
            arr_tmp = np.array(lst_tmp)
            # lst_rst.append([*lst_tmp])
            lst_rst.append(arr_tmp)
        else:  # 如果当前还不是最底层则Y 轴+1 继续往下递归，所以递归最大层数就是Y 轴的最大值
               # lst_rst 和lst_tmp 的地址也传到下次递归中，这样不论在哪一层中修改的都是同一个list 对象
            dynloop_rcsn(data, cur_y_idx+1, lst_rst, lst_tmp)
        lst_tmp.pop()  # 在本次循环最后，不管是递归回来的，还是最底层循环的，都要将lst_tmp 最后一个元素移除

    return lst_rst

def Get_Order_List(n_views, n_balls):
    # first view: the order of balls
    ori_order = [i for i in range(n_balls)]
    # calculate all possible order
    resid = []
    perm(ori_order, 0, len(ori_order), resid)
    num_order = len(resid)
    data = [[ori_order]]
    for i in range(n_views - 1):
        data.append(resid)
    index_list = dynloop_rcsn(data)
    return index_list