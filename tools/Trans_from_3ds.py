#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) 2023 -2023 WangHaoyu.All Rights Reserved
import numpy as np
def Threepps2Tran(kps0_init,kps1_init):
    center0 = np.mean(kps0_init,0,keepdims=True)
    center1 = np.mean(kps1_init,0,keepdims=True)
    m = (kps1_init-center1).T @ (kps0_init-center0)
    U,S,VT = np.linalg.svd(m)
    rotation = VT.T @ U.T #predicted RT
    offset = center0 - (center1 @ rotation.T)
    transform=np.concatenate([rotation,offset.T],1)
    return transform #3*4