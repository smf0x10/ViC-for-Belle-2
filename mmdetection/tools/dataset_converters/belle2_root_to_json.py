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

from tools.dataset_converters.root_to_utils import *


def str_to_floats(s, default_return = None):
    """
    Convert the string s to a floating-point list. 
    If it does not contain any floating-point numbers, 
    Zero is returned.
    """
    num_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']     # Numeric characters
    i_begin = 0                                                             # String beginning indicator
    i_end = 0                                                               # String end indicator
    floats = []

    s += '_'                                                                # Make s always end with a non-numeric character to ensure that the last digit can be output

    for i, c in enumerate(s):
        if c in num_chars:                                                  # If it's a numeric character
            i_end = i + 1                                                   # Push the end indicator back one notch!
        else:                                                               # Otherwise it is not a numeric character
            if i_end > i_begin:                                             # In this case, if the length of the valid string is greater than 0
                floats.append(float(s[i_begin:i_end]))                      # Convert the string to floating-point numbers and store it in bbox_scales list
            i_begin = i + 1                                                 # Push the start indicator back one notch!

    if len(floats) == 0: floats = default_return
    print("floats:", floats)
    return floats


def strides_and_ratios_to_wh(strides, ratios):
    """
    Convert strides and ratios to numpy vectors of shape (len_strides*len_ratios, 2). 
    If strides is None, an all-zero numpy vector of shape (1, 2) is returned.
    """
    w = np.array([0.0])
    h = np.array([0.0])

    if strides is not None:
        w = (w[:, np.newaxis] + np.array(strides)).flatten()
        h = (h[:, np.newaxis] + np.array(strides)).flatten()

        if ratios is not None:
            h_ratios = np.sqrt(np.array(ratios))
            w_ratios = 1 / h_ratios
            w = (w[:, np.newaxis] * w_ratios).flatten()
            h = (h[:, np.newaxis] * h_ratios).flatten()

    wh = np.hstack((w[:, np.newaxis], h[:, np.newaxis]))
    print("len(wh):", len(wh))
    for single_wh in wh: print(single_wh)
    return wh


def root_to_json(
    srcfile: str,
    destroot: str,
    split_size: int = 100000,
    width: int = 960,
    height: int = 480,
    easy_scale: float = 10.0,
    # 
    strides: str = "",
    ratios: str = "",
    scales: str = "10.0",
):
    t1 = datetime.datetime.now()

    tree = uproot.open(srcfile + ":tree") # Set the tree to open here

    t2 = datetime.datetime.now()
    print("[uproot] : open file successfully! time: {}".format(t2 - t1))

    data = tree.arrays(library="np")

    t3 = datetime.datetime.now()
    print("[numpy]  : read data successfully! time: {}".format(t3 - t2))

    # for key, value in data.items():
    #     print(key, len(value))

    flag_cc_array = data['flag_cc'] # flag_cc: used to set category_id in final json
    flag_SB_array = data['flag_SB'] # flag_SB: used to set category_id in final json

    p_RM_array = data['mcP'] # p_RM: Momentum? Or radius?
    phi_RM_array = data['clusterPhi'] # phi_RM: phi angle of each event
    the_RM_array = data['clusterTheta'] # the_RM: theta angle of each event

    n_hit_array = data['clusterNHits'] # n_hit: Number of hits in each event
    m_eng_arrays = data['m_eng'] # m_eng
    m_phi_arrays = data['m_phi'] # m_phi
    m_the_arrays = data['m_the'] # m_the
    m_time_arrays = data['m_time'] # m_time:                                # The current value set is {0.0, 1.0, 2.0,..., 20.0}

    num_event = len(n_hit_array)
    print("num_event:", num_event)

    split_num = (num_event // split_size) + 1

    _path, _file = os.path.split(srcfile)                                   # Split folders and files
    _filename, _fileext = os.path.splitext(_file)                           # Split the file name from the file extension. All you need is the file name at the end!

    # The folder name of the json file
    bbox_scales_str = "bbox_scale"
    if easy_scale > 0.0:
        bbox_scales_str += ('_' + str(int(easy_scale)))
    else:
        _strides = str_to_floats(strides, None)
        _ratios = str_to_floats(ratios, None)
        _scales = str_to_floats(scales, [10.0])
        _wh = strides_and_ratios_to_wh(_strides, _ratios)
        for single_scale in _scales: bbox_scales_str += ('_' + str(int(single_scale)))
    destfolder = os.path.join(destroot, bbox_scales_str)

    if not os.path.exists(destfolder):
        os.makedirs(destfolder)

    # Note: JSON numbers are counted from 1. 
    # single_image['file_name'], single_image['id'], single_obj['image_id'], single_obj['id'], 
    # and the begin-end naming of JSON files are all image_id decimal 8 digits.
    image_i = 0
    ann_i = 0
    data_dict_checked = False

    for j in range(split_num):                                              # For each json file that is about to be generated

        # json文件的基础信息
        data_dict = {}
        info = {"description": "HEP2COCO 2024 Dataset",
                "url": "",
                "version": "0.01b",
                "year": 2024,
                "contributor": "Hongtian Yu, Yangu Li",
                "date_created": "2024/11/01"}
        categories = [
            {'id': 1, 'name': 'Nm', 'supercategory': 'Nm'},                 # flag_cc: (-1); flag_SB: (1): (-1) * 0.5 + (1) * 2 - 0.5 = 1
            {'id': 2, 'name': 'Np', 'supercategory': 'Np'},                 # flag_cc: (+1); flag_SB: (1): (+1) * 0.5 + (1) * 2 - 0.5 = 2
            # {'id': 3, 'name': 'Lmdm', 'supercategory': 'Lmdm'},
            # {'id': 4, 'name': 'Lmdp', 'supercategory': 'Lmdp'},
            ]
        data_dict['info'] = info
        data_dict['categories'] = categories
        data_dict['images'] = []
        data_dict['annotations'] = []

        # json file
        i_begin = j * split_size
        i_end = min(num_event, (j + 1) * split_size)
        print("i_begin: {}, i_end: {}".format(i_begin, i_end))              # 0~100000, 100000~200000, ...

        for i in tqdm(range(i_begin, i_end)):                               # For each image in the json file

            # Note that you must perform a forced type conversion first, otherwise json.dump() will report an error of unsupported data types
            flag_cc_i = int(flag_cc_array[i])
            flag_SB_i = int(flag_SB_array[i])

            p_RM_i = float(p_RM_array[i])
            phi_RM_i = float(phi_RM_array[i])
            the_RM_i = float(the_RM_array[i])

            n_hit_i = int(n_hit_array[i])
            m_eng_array = m_eng_arrays[i].astype(float)
            m_phi_array = m_phi_arrays[i].astype(float)
            m_the_array = m_the_arrays[i].astype(float)
            m_time_array = m_time_arrays[i].astype(int)

            category_id = int(flag_cc_i * 0.5 + flag_SB_i * 2.0 - 0.499)

            x_ctr_array, y_ctr_array, w_array, h_array, xmin_array, ymax_array = phithe_to_xywh_np(m_phi_array, m_the_array, width=width, height=height)
            xmax_array = xmin_array + w_array
            ymin_array = ymax_array - h_array
            xyxy_array = np.vstack([xmin_array, ymin_array, xmax_array, ymax_array]).T

            image_i += 1

            single_image = {}
            single_image['file_name'] = _filename + "_{:08}".format(image_i) + ".png"
            single_image['id'] = image_i
            single_image['width'] = width
            single_image['height'] = height
            single_image['n_hit'] = n_hit_i
            single_image['m_eng'] = m_eng_array.tolist()
            # single_image['m_phi'] = m_phi_array.tolist()
            # single_image['m_the'] = m_the_array.tolist()
            single_image['xyxy'] = xyxy_array.astype(int).tolist()
            single_image['m_time'] = m_time_array.tolist()
            data_dict['images'].append(single_image)

            if easy_scale > 0.0:
                x_ctr_float, y_ctr_float, w_int, h_int, _r5, _r6 = phithe_to_xywh_np(phi_RM_i, the_RM_i, width=width, height=height)

                w_ex = w_int * easy_scale
                h_ex = h_int * easy_scale
                xmin_ex = x_ctr_float - w_ex * 0.5
                xmax_ex = x_ctr_float + w_ex * 0.5
                ymin_ex = y_ctr_float - h_ex * 0.5
                ymax_ex = y_ctr_float + h_ex * 0.5
                area_ex = w_ex * h_ex

                x_inbbox = (np.abs(x_ctr_array - x_ctr_float) < w_ex)
                y_inbbox = (np.abs(y_ctr_array - y_ctr_float) < h_ex)
                xy_inbbox = np.vstack([x_inbbox, y_inbbox])
                inbbox = np.all(xy_inbbox, axis=0)

                total_eng = np.sum(m_eng_array)
                bbox_eng = np.sum(m_eng_array[inbbox])
                btr_eng = bbox_eng / total_eng

                ann_i += 1

                single_obj = {}
                single_obj['area'] = area_ex
                single_obj['category_id'] = category_id
                single_obj['segmentation'] = [[xmin_ex, ymin_ex, 
                                               xmax_ex, ymin_ex, 
                                               xmax_ex, ymax_ex, 
                                               xmin_ex, ymax_ex]]
                single_obj['iscrowd'] = 0
                single_obj['bbox'] = xmin_ex, ymin_ex, w_ex, h_ex
                single_obj['image_id'] = image_i
                single_obj['id'] = ann_i
                single_obj['p_RM'] = p_RM_i
                single_obj['phi_RM'] = phi_RM_i
                single_obj['the_RM'] = the_RM_i
                single_obj['btr_eng'] = btr_eng
                data_dict['annotations'].append(single_obj)

            else:                                                           # Ablation experiments designed for fixed-size pseudo-frames are generally not used
                for single_wh in _wh:                                       # For each box in the picture
                    single_w, single_h = single_wh

                    for single_scale in _scales:
                        x_ctr_float, y_ctr_float, w_int, h_int, _r5, _r6 = phithe_to_xywh_np(phi_RM_i, the_RM_i, width=width, height=height)

                        w_ex = (single_w * single_scale) if (single_w > 0.0) else (w_int * single_scale)
                        h_ex = (single_h * single_scale) if (single_h > 0.0) else (h_int * single_scale)
                        xmin_ex = x_ctr_float - w_ex * 0.5
                        xmax_ex = x_ctr_float + w_ex * 0.5
                        ymin_ex = y_ctr_float - h_ex * 0.5
                        ymax_ex = y_ctr_float + h_ex * 0.5
                        area_ex = w_ex * h_ex

                        ann_i += 1

                        single_obj = {}
                        single_obj['area'] = area_ex
                        single_obj['category_id'] = category_id
                        single_obj['segmentation'] = [[xmin_ex, ymin_ex, 
                                                       xmax_ex, ymin_ex, 
                                                       xmax_ex, ymax_ex, 
                                                       xmin_ex, ymax_ex]]
                        single_obj['iscrowd'] = 0
                        single_obj['bbox'] = xmin_ex, ymin_ex, w_ex, h_ex
                        single_obj['image_id'] = image_i
                        single_obj['id'] = ann_i
                        single_obj['p_RM'] = p_RM_i
                        single_obj['phi_RM'] = phi_RM_i
                        single_obj['the_RM'] = the_RM_i
                        single_obj['btr_eng'] = 1.0
                        data_dict['annotations'].append(single_obj)

            if not data_dict_checked:
                # # Check the values
                print(data_dict)
                # Check the variable type
                # print("\nsingle_image:")
                # for key, value in single_image.items(): print(key, type(value))
                # print("\nsingle_obj:")
                # for key, value in single_obj.items(): print(key, type(value))
                # print("")
                # # Check the visualization
                # visualization(single_image=data_dict['images'][0], gts=data_dict['annotations'])
                data_dict_checked = True

        # The file name of the json file
        destfile = os.path.join(destfolder, _filename 
                                + "__b{:08}".format(i_begin + 1) 
                                + "__e{:08}".format(i_end) 
                                + ".json")
        t4 = datetime.datetime.now()

        with open(destfile, 'w') as f_out:
            json.dump(data_dict, f_out)

        t5 = datetime.datetime.now()
        print("[json]   : write to \"{}\" successfully! time: {}".format(destfile, t5 - t4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcfile", type = str, default = "./data/BESIII_training_sample/Nm_1m.root", help = "srcfile")
    parser.add_argument("--destroot", type = str, default = "./data/HEP2COCO/", help = "destroot")
    parser.add_argument("--split_size", type = int, default = 100000, help = "split size")
    parser.add_argument("--width", type = int, default = 960, help = "width")
    parser.add_argument("--height", type = int, default = 480, help = "height")
    parser.add_argument("--easy_scale", type = float, default = 10.0, help = "easy scale")
    # 
    parser.add_argument("--strides", type = str, default = "", help = "strides")
    parser.add_argument("--ratios", type = str, default = "", help = "ratios")
    parser.add_argument("--scales", type = str, default = "10.0", help = "scales")
    # 
    opt = parser.parse_args()

    root_to_json(
        srcfile = opt.srcfile,
        destroot = opt.destroot,
        split_size = opt.split_size,
        width = opt.width,
        height = opt.height,
        easy_scale = opt.easy_scale,
        # 
        strides = opt.strides,
        ratios = opt.ratios,
        scales = opt.scales,
    )

