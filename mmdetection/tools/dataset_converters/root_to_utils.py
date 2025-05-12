import os
import argparse
import json
import datetime
from tqdm import tqdm
from typing import Optional, List

import math
import numpy as np
import uproot
import mmcv


# def centercroprescale(img, w_scale, h_scale):
#     h_img, w_img, c = img.shape
# 
#     w_scale_new = int(max(w_scale, w_img * h_scale / h_img))
#     h_scale_new = int(max(h_scale, h_img * w_scale / w_img))
# 
#     img_np = mmcv.imresize(img, (w_scale_new, h_scale_new))
# 
#     xmin = int((w_scale_new - w_scale) * 0.5)
#     ymin = int((h_scale_new - h_scale) * 0.5)
#     img_np = img_np[ymin:ymin+h_scale, xmin:xmin+w_scale, :]
# 
#     return img_np


def create_bg_np(
    h, w, c,
    with_time: int = 0,
    bg_version: Optional[str] = None,
    snr_db: float = 10.0,
    # json_path: Optional[str] = None,
) -> np.ndarray:
    # mean & std from ImageNet:
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    # rgb to bgr
    mean.reverse()
    std.reverse()

    snr = 10 ** (snr_db / 20)
    seed = 0

    if bg_version == 'black_randn':
        img_np = np.random.randn(h, w, c) * np.array(std) / snr
    elif bg_version == 'black_randn_seed':
        np.random.seed(seed)
        img_np = np.random.randn(h, w, c) * np.array(std) / snr
    # elif bg_version == 'image_randint':
    #     with open(json_path, 'r') as f_in:
    #         image_info = json.load(f_in)
    #     image_root = image_info['image_root']
    #     image_list = image_info['image_list']
    #     num_images = image_info['num_images']
    #     ind = np.random.randint(0, num_images)
    #     img_np = mmcv.imread(os.path.join(image_root, image_list[ind]))
    #     img_np = centercroprescale(img_np, w, h)
    #     img_np = img_np / snr
    elif bg_version == 'white':
        img_np = np.zeros((h, w, c)) + 255
    else:
        img_np = np.zeros((h, w, c))

    if with_time > 1:
        img_np = np.concatenate([img_np, np.zeros((h, w, 1))], axis=-1)

    return img_np


def eng_to_rgb_np(eng):
    # if not isinstance(eng, np.ndarray):
    #     eng = np.array([eng])

    # This gets run to convert m_eng from the annotation file to an rgb value
    eng_log10 = np.log10(eng)

    index_low =            (eng_log10 <  -2.3)                                          # 小于5e-3 GeV
    index_mid = np.vstack([(eng_log10 >= -2.3), (eng_log10 <  -1.3)]).all(axis=0)
    index_high =                                (eng_log10 >= -1.3)                     # 大于5e-2 GeV

    r = np.zeros_like(eng)
    g = np.zeros_like(eng)
    b = np.zeros_like(eng)

    if index_low.any():
        rgb_norm =   np.clip((eng_log10[index_low] + 3.3), a_min=0, a_max=1) ** 0.5     # [-3.3, -2.3) -> [0, 1)
        b[index_low] = rgb_norm * 255 + 1
    if index_mid.any():
        rgb_norm =           (eng_log10[index_mid] + 2.3)                    ** 0.6     # [-2.3, -1.3) -> [0, 1)
        g[index_mid] = rgb_norm * 255 + 1
    if index_high.any():
        #                                    in rad: np.arctan(3) = 1.2490457723982544
        rgb_norm = np.arctan((eng_log10[index_high] + 1.3) * 2.5) / 1.2490457723982544  # [-1.3, -0.1) -> [0, 1)
        r[index_high] = rgb_norm * 255 + 1

    return r, g, b


def load_rgb(
    single_image: dict,
    with_time: int = 0,
    bg_version: Optional[str] = None,
    snr_db: float = 10.0,
    # json_path: Optional[str] = None,
) -> np.ndarray:
    """
    加载RGB值。要求：
    single_image具有这些键: 'height', 'width', 'n_hit', 'm_eng', 'xyxy', 'm_time'
    """

    h = single_image['height']
    w = single_image['width']
    c = 3  # B, G, R

    img_np = create_bg_np(
        h, w, c, 
        with_time, 
        bg_version, snr_db, 
    )

    n_hit = single_image['n_hit']
    m_eng = single_image['m_eng']
    xyxy = single_image['xyxy']
    m_time = single_image['m_time']

    r_array, g_array, b_array = eng_to_rgb_np(np.array(m_eng))  # Here `m_eng` is a list!

    for i in range(n_hit):
        r, g, b = r_array[i], g_array[i], b_array[i]
        xmin, ymin, xmax, ymax = xyxy[i]
        img_np[ymin:ymax, xmin:xmax, :3] = np.array([b, g, r])
        if with_time > 1:
            img_np[ymin:ymax, xmin:xmax, 3] = m_time[i] + 1     # {0, 1, 2, ..., 20} -> {1, 2, 3, ..., 21}

    return img_np

def load_rgb_klm(
    single_image: dict,
    #ecl_image: dict,
    #klm1_image: dict,
    #klm2_image: dict,
    #klm3_image: dict,
    with_time: int = 0,
    bg_version: Optional[str] = None,
    snr_db: float = 10.0
) -> np.ndarray:
    """
        Added by Sean Frett
        Converts the dictionaries of the ECL and KLM data into one mosaic image
    """
    ecl_image = single_image['ecl']
    klm1_image = single_image['klm1']
    klm2_image = single_image['klm2']
    klm3_image = single_image['klm3']
    h = single_image['height']
    w = single_image['width']
    c = 3  # B, G, R

    img_np = create_bg_np(
        h*4, w, c,
        with_time,
        bg_version, snr_db,
    )

    n_hit_ecl = ecl_image['n_hit']
    m_eng_ecl = ecl_image['m_eng']
    xyxy_ecl = ecl_image['xyxy']
    m_time_ecl = ecl_image['m_time']

    r_array_ecl, g_array_ecl, b_array_ecl = eng_to_rgb_np(np.array(m_eng_ecl))  # Here `m_eng` is a list!

    n_hit_klm1 = klm1_image['n_hit']
    m_eng_klm1 = klm1_image['m_eng']
    xyxy_klm1 = klm1_image['xyxy']
    m_time_klm1 = klm1_image['m_time']

    r_array_klm1, g_array_klm1, b_array_klm1 = eng_to_rgb_np(np.array(m_eng_klm1))

    n_hit_klm2 = klm2_image['n_hit']
    m_eng_klm2 = klm2_image['m_eng']
    xyxy_klm2 = klm2_image['xyxy']
    m_time_klm2 = klm2_image['m_time']

    r_array_klm2, g_array_klm2, b_array_klm2 = eng_to_rgb_np(np.array(m_eng_klm2))

    n_hit_klm3 = klm3_image['n_hit']
    m_eng_klm3 = klm3_image['m_eng']
    xyxy_klm3 = klm3_image['xyxy']
    m_time_klm3 = klm3_image['m_time']

    r_array_klm3, g_array_klm3, b_array_klm3 = eng_to_rgb_np(np.array(m_eng_klm3))

    for i in range(n_hit_ecl):
        r, g, b = r_array_ecl[i], g_array_ecl[i], b_array_ecl[i]
        xmin, ymin, xmax, ymax = xyxy_ecl[i]
        img_np[ymin:ymax, xmin:xmax, :3] = np.array([b, g, r])
        if with_time > 1:
            img_np[ymin:ymax, xmin:xmax, 3] = m_time[i] + 1 # Not even sure if this part will get used or not, but better to keep changes consistent

    for i in range(n_hit_klm1):
        r, g, b = r_array_klm1[i], g_array_klm1[i], b_array_klm1[i]
        xmin, ymin, xmax, ymax = xyxy_klm1[i]
        img_np[ymin + h:ymax + h, xmin:xmax, :3] = np.array([b, g, r])
        if with_time > 1:
            img_np[ymin+h:ymax+h, xmin:xmax, 3] = m_time[i] + 1

    for i in range(n_hit_klm2):
        r, g, b = r_array_klm2[i], g_array_klm2[i], b_array_klm2[i]
        xmin, ymin, xmax, ymax = xyxy_klm2[i]
        img_np[ymin + (2*h):ymax + (2*h), xmin:xmax, :3] = np.array([b, g, r])
        if with_time > 1:
            img_np[ymin + (2*h):ymax + (2*h), xmin:xmax, 3] = m_time[i] + 1

    for i in range(n_hit_klm3):
        r, g, b = r_array_klm3[i], g_array_klm3[i], b_array_klm3[i]
        xmin, ymin, xmax, ymax = xyxy_klm3[i]
        img_np[ymin + (3*h):ymax + (3*h), xmin:xmax, :3] = np.array([b, g, r])
        if with_time > 1:
            img_np[ymin + (3*h):ymax + (3*h), xmin:xmax, 3] = m_time[i] + 1


    return img_np



def visualization(
    single_image: dict,
    gts: List[dict] = None,
    single_pred: dict = None,
    output_dir: str = "./",
    with_hint: bool = True,
    klm: bool = False
) -> np.ndarray:
    """可视化函数（图像保存到本地）
    要求：
    single_image具有这些键: 'file_name', 'height', 'width', 'n_hit', 'm_eng', 'xyxy'
    gts的列表元素具有这些键: 'p_RM', 'bbox'
    single_pred具有这些键: 'pred_instances.bboxes'
    """
    file_name = single_image['file_name']  # e.g. 'Nm_1m_00000001.png'

    t1 = datetime.datetime.now()

    if klm:
        img_np = load_rgb_klm(single_image, bg_version='white')
    else:
        img_np = load_rgb(single_image, bg_version='white')

    t2 = datetime.datetime.now()

    mmcv.imwrite(img_np, output_dir + "raw_" + file_name)
    if with_hint: print("[numpy]  : write BGR value successfully! time: {}".format(t2 - t1))

    # 可视化gt框
    if isinstance(gts, list) and len(gts) > 0:
        gt_eng = gts[0]['p_RM']
        output_img_path = output_dir + "gt_{:04}MeV_".format(int(gt_eng * 1000)) + file_name

        bboxes_list = []
        for gt in gts:
            x1, y1, w1, h1 = gt['bbox']
            bboxes_list.append([x1, y1, x1 + w1, y1 + h1])
        bboxes = np.array(bboxes_list)

        t3 = datetime.datetime.now()

        # https://github.com/open-mmlab/mmcv/blob/main/mmcv/visualization/image.py
        img_np = mmcv.imshow_bboxes(
            img = img_np,
            bboxes = bboxes,
            # colors = (0, 0, 255),
            colors = 'red',
            # top_k: int = -1,
            thickness = 2,
            show = False,
            # win_name: str = '',
            # wait_time: int = 0,
            out_file = output_img_path,
        )

        t4 = datetime.datetime.now()
        if with_hint: print("[mmcv]   : write to \"{}\" successfully! time: {}".format(output_img_path, t4 - t3))

    # 可视化pred框
    if single_pred is not None:
        pred_eng = single_pred["pred_instances"].get("engs", [0.0])
        output_img_path = output_dir + "pred_{:04}MeV_".format(int(pred_eng[0] * 1000)) + file_name

        single_pred_bboxes = single_pred["pred_instances"]["bboxes"]
        bboxes = np.array(single_pred_bboxes)

        t5 = datetime.datetime.now()

        # https://github.com/open-mmlab/mmcv/blob/main/mmcv/visualization/image.py
        img_np = mmcv.imshow_bboxes(
            img = img_np,
            bboxes = bboxes,
            # colors = (255, 0, 0),
            colors = 'blue',
            # top_k: int = -1,
            thickness = 2,
            show = False,
            # win_name: str = '',
            # wait_time: int = 0,
            out_file = output_img_path,
        )

        t6 = datetime.datetime.now()
        if with_hint: print("[mmcv]   : write to \"{}\" successfully! time: {}".format(output_img_path, t6 - t5))

    return img_np


def phithe_to_xywh_np(phi, the, width=960, height=480):
    """Turn (phi, the) to (x_ctr, y_ctr, w, h) with numpy

    params:
        phi :           float or 1D numpy.ndarray
        the :           float or 1D numpy.ndarray
        width :         int, default = 960
        height :        int, default = 480

    return:
        x_ctr :         float or 1D numpy.ndarray
        y_ctr :         float or 1D numpy.ndarray
        w_cell :        int or 1D numpy.ndarray
        h_cell :        int or 1D numpy.ndarray
        xmin_cell :     int or 1D numpy.ndarray
        ymax_cell :     int or 1D numpy.ndarray
    """
    phi_np_flag = isinstance(phi, np.ndarray)
    the_np_flag = isinstance(the, np.ndarray)
    assert phi_np_flag == the_np_flag

    # 先转换成1D矢量
    if not phi_np_flag:
        phi = np.array([phi])
        the = np.array([the])
    # 再判断越界
    assert phi.ndim == 1 and np.vstack([phi >= -np.pi, phi < np.pi]).all()
    assert the.ndim == 1 and np.vstack([the >= 0, the < np.pi]).all()

    half_width = width * 0.5
    x_ctr = phi / np.pi * half_width + half_width     # float or 1D numpy.ndarray
    y_ctr = the / np.pi * height                      # 1D numpy.ndarray

    w_px = np.array([
        30, 30, 24, 24, 20, 20,         # empty
        20,                             # empty
        15, 15, 12, 12, 10, 10, 
        10,                             # empty
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        10,                             # empty
        10, 10, 12, 12, 15, 15, 
        20,                             # empty
        20, 20, 24, 24, 30, 30,         # empty
    ])
    h_px = np.array([
        8, 8, 8, 8, 7, 7,               # empty
        7,                              # empty
        6, 6, 6, 6, 5, 5, 
        5,                              # empty
        5, 5, 5, 5, 5, 
        6, 6, 6, 6, 
        7, 7, 7, 7, 7, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        7, 7, 7, 7, 7, 
        6, 6, 6, 6, 
        5, 5, 5, 5, 5, 
        5,                              # empty
        5, 5, 6, 6, 6, 6, 
        7,                              # empty
        7, 7, 8, 8, 8, 8,               # empty
    ])
    hh_px = np.array([
        8, 16, 24, 32, 39, 46, 
        53, 
        59, 65, 71, 77, 82, 87, 
        92, 
        97, 102, 107, 112, 117, 
        123, 129, 135, 141, 
        148, 155, 162, 169, 176, 
        184, 192, 200, 208, 216, 224, 232, 240, 
        248, 256, 264, 272, 280, 288, 296, 304, 
        311, 318, 325, 332, 339, 
        345, 351, 357, 363, 
        368, 373, 378, 383, 388, 
        393, 
        398, 403, 409, 415, 421, 427, 
        434, 
        441, 448, 456, 464, 472, 480, 
    ])

    y_ctr_2D = y_ctr[:, np.newaxis]     # 转换为2D矢量，shape (:, 1)
    hh_px_2D = hh_px[np.newaxis, :]     # 转换为2D矢量，shape (1, :)
    ind = np.sum((y_ctr_2D - hh_px_2D) >= 0, axis=1)
    w_cell = w_px[ind]
    h_cell = h_px[ind]
    xmin_cell = (x_ctr / w_cell).astype(int) * w_cell
    ymax_cell = hh_px[ind]

    if not phi_np_flag:                 # 转换回浮点数/整数！
        x_ctr = float(x_ctr[0])
        y_ctr = float(y_ctr[0])
        w_cell = int(w_cell[0])
        h_cell = int(h_cell[0])
        xmin_cell = int(xmin_cell[0])
        ymax_cell = int(ymax_cell[0])

    return x_ctr, y_ctr, w_cell, h_cell, xmin_cell, ymax_cell

