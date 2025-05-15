import os
import argparse
import json
import datetime
from tqdm import tqdm
from typing import Optional, List

import csv
import math
import numpy as np
import uproot
# import mmcv

from root_to_utils import *
from cell_position_lookup import *
from klm_positions import klm_hit_position


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

def cut_energy(E_cut, E_dep, cell_id, time):
    ret = ([], [], [])
    for i in range(len(E_dep)):
        if (E_dep[i] >= E_cut):
            ret[0].append(E_dep[i])
            ret[1].append(cell_id[i])
            ret[2].append(time[i])
    return ret

def cell_id_to_phi_theta(cell_ids):
    """
    Gets the (phi, theta) coordinates in radians of the given array of cell ids
    """
    # print(cell_ids)
    ret_theta = []
    ret_phi = []
    for cell_id in cell_ids:
        theta = cell_position[cell_id-1][1]
        phi = cell_position[cell_id-1][0]
        ret_theta.append(theta)
        ret_phi.append(phi)
    return (ret_phi, ret_theta)

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

    data = tree

    t3 = datetime.datetime.now()
    print("[numpy]  : read data successfully! time: {}".format(t3 - t2))

    _path, _file = os.path.split(srcfile)                                   # Split folders and files
    _filename, _fileext = os.path.splitext(_file)                           # Split the file name from the file extension. All you need is the file name at the end!

    mc_particles_mass = data["MCParticles"]["MCParticles.m_mass"].array().tolist()
    mc_particles_p_x = data["MCParticles"]["MCParticles.m_momentum_x"].array().tolist()
    mc_particles_p_y = data["MCParticles"]["MCParticles.m_momentum_y"].array().tolist()
    mc_particles_p_z = data["MCParticles"]["MCParticles.m_momentum_z"].array().tolist()

    klm_cluster_x = data["KLMClusters"]["KLMClusters.m_globalX"].array().tolist()
    klm_cluster_y = data["KLMClusters"]["KLMClusters.m_globalY"].array().tolist()
    klm_cluster_z = data["KLMClusters"]["KLMClusters.m_globalZ"].array().tolist()
    klm_cluster_p = data["KLMClusters"]["KLMClusters.m_p"].array().tolist()

    klm_hit_layer = data["KLMHit2ds"]["KLMHit2ds.m_Layer"].array().tolist()
    klm_hit_x = data["KLMHit2ds"]["KLMHit2ds.m_GlobalX"].array().tolist()
    klm_hit_y = data["KLMHit2ds"]["KLMHit2ds.m_GlobalY"].array().tolist()
    klm_hit_z = data["KLMHit2ds"]["KLMHit2ds.m_GlobalZ"].array().tolist()
    klm_hit_eng = data["KLMHit2ds"]["KLMHit2ds.m_EnergyDeposit"].array().tolist()
    klm_hit_subdetector = data["KLMHit2ds"]["KLMHit2ds.m_Subdetector"].array().tolist()
    klm_hit_section = data["KLMHit2ds"]["KLMHit2ds.m_Section"].array().tolist()
    klm_hit_sector = data["KLMHit2ds"]["KLMHit2ds.m_Sector"].array().tolist()
    klm_hit_strip = data["KLMHit2ds"]["KLMHit2ds.m_Strip[2]"].array().tolist()
    klm_hit_last_strip = data["KLMHit2ds"]["KLMHit2ds.m_LastStrip[2]"].array().tolist()

    #print("lengths:")
    #print(len(mc_particles_mass))
    #print(len(mc_particles_p_x))
    #print(len(mc_particles_p_y))
    #print(len(mc_particles_p_z))

    ecl_hit_Edep = data["ECLHits"]["ECLHits.m_Edep"].array().tolist()
    ecl_hit_CellId = data["ECLHits"]["ECLHits.m_CellId"].array().tolist()
    ecl_hit_time = data["ECLHits"]["ECLHits.m_TimeAve"].array().tolist()

    #print(len(ecl_hit_Edep))
    #print(len(ecl_hit_CellId))
    #print(len(ecl_hit_time))

    num_event = len(mc_particles_mass)
    hit_info = [cut_energy(0.005, ecl_hit_Edep[i], ecl_hit_CellId[i], ecl_hit_time[i]) for i in range(num_event)]
    print("Pruning events with 0 hits...")
    for i in range(num_event-1, -1, -1):
        if len(hit_info[i][0]) == 0 or len(klm_cluster_p[i]) == 0:
            num_event -= 1
            hit_info.pop(i)
            mc_particles_mass.pop(i)
            mc_particles_p_x.pop(i)
            mc_particles_p_y.pop(i)
            mc_particles_p_z.pop(i)
            ecl_hit_Edep.pop(i)
            ecl_hit_CellId.pop(i)
            ecl_hit_time.pop(i)
            klm_cluster_x.pop(i)
            klm_cluster_y.pop(i)
            klm_cluster_z.pop(i)
            klm_cluster_p.pop(i)
            klm_hit_layer.pop(i)
            klm_hit_x.pop(i)
            klm_hit_y.pop(i)
            klm_hit_z.pop(i)
            klm_hit_eng.pop(i)
            klm_hit_subdetector.pop(i)
            klm_hit_sector.pop(i)
            klm_hit_section.pop(i)
            klm_hit_strip.pop(i)
            klm_hit_last_strip.pop(i)

    p_cluster_array = np.array([p[0] for p in klm_cluster_p])
    phi_cluster_array = np.array([cartesian_to_polar(klm_cluster_x[i][0], klm_cluster_y[i][0]) for i in range(num_event)])
    the_cluster_array = np.array([cartesian_to_polar(klm_cluster_z[i][0], math.sqrt(klm_cluster_x[i][0]**2 + klm_cluster_y[i][0]**2)) for i in range(num_event)])

    p_RM_array = np.array([math.sqrt(mc_particles_p_x[i][0]**2 + mc_particles_p_y[i][0]**2 + mc_particles_p_z[i][0]**2) for i in range(num_event)])
    phi_RM_array = np.array([cartesian_to_polar(mc_particles_p_x[i][0], mc_particles_p_y[i][0]) for i in range(num_event)])
    the_RM_array = np.array([cartesian_to_polar(mc_particles_p_z[i][0], math.sqrt(mc_particles_p_x[i][0]**2 + mc_particles_p_y[i][0]**2)) for i in range(num_event)])
    n_hit_array = np.array([len(hit_info[i][0]) for i in range(num_event)])


    with open(destroot + "/" + _filename + "_klm_cluster.csv", "w") as csvfile:
        print("writing csv...")
        writer = csv.DictWriter(csvfile, fieldnames=["p_cluster", "phi_cluster", "theta_cluster", "p_MC", "phi_MC", "theta_MC"])
        writer.writeheader()
        for i in range(len(p_cluster_array)):
            writer.writerow({"p_cluster": p_cluster_array[i], "phi_cluster": phi_cluster_array[i], "theta_cluster": the_cluster_array[i],
                             "p_MC": p_RM_array[i], "phi_MC": phi_RM_array[i], "theta_MC": the_RM_array[i]})

    m_eng_arrays = [np.array(hit_info[i][0]) for i in range(num_event)]
    # print(m_eng_arrays)
    m_phi_arrays = [np.array(cell_id_to_phi_theta(hit_info[i][1])[0]) for i in range(num_event)]
    # print(m_phi_arrays)
    m_the_arrays = [np.array(cell_id_to_phi_theta(hit_info[i][1])[1]) for i in range(num_event)]
    # print(m_the_arrays)
    m_time_arrays = [np.array(hit_info[i][2]) for i in range(num_event)]

    m_klm1_eng_arrays = []
    m_klm2_eng_arrays = []
    m_klm3_eng_arrays = []

    m_klm1_phi_arrays = []
    m_klm2_phi_arrays = []
    m_klm3_phi_arrays = []

    m_klm1_the_arrays = []
    m_klm2_the_arrays = []
    m_klm3_the_arrays = []

    klm1_n_hit_array = [0]*num_event
    klm2_n_hit_array = [0]*num_event
    klm3_n_hit_array = [0]*num_event

    for e in range(num_event):
        klm_layer_1_indexes = []
        klm_layer_2_indexes = []
        klm_layer_3_indexes = []
        for i in range(len(klm_hit_layer[e])):
            if (klm_hit_layer[e][i] == 1):
                klm_layer_1_indexes.append(i)
                klm1_n_hit_array[e] += 1
            elif (klm_hit_layer[e][i] == 2):
                klm_layer_2_indexes.append(i)
                klm2_n_hit_array[e] += 1
            elif (klm_hit_layer[e][i] == 3):
                klm_layer_3_indexes.append(i)
                klm3_n_hit_array[e] += 1

#        for i in klm_layer_3_indexes:
#            if klm_hit_strip[e][i][0] == klm_hit_last_strip[e][i][0] and klm_hit_strip[e][i][1] == klm_hit_last_strip[e][i][1] and klm_hit_subdetector[e][i] == 1:
#                hit2d_positions = (cartesian_to_polar(klm_hit_x[e][i], klm_hit_y[e][i]), cartesian_to_polar(klm_hit_z[e][i], math.sqrt(klm_hit_x[e][i]**2 + klm_hit_y[e][i]**2)))
#                my_mapping = klm_hit_position(klm_hit_layer[e][i], klm_hit_sector[e][i], klm_hit_subdetector[e][i], klm_hit_section[e][i], klm_hit_strip[e][i][0], klm_hit_strip[e][i][1])
#                print("strips", klm_hit_strip[e][i])
#                print("off by:", my_mapping[0] - hit2d_positions[0], my_mapping[1] - hit2d_positions[1])
#                if abs(my_mapping[0] - hit2d_positions[0]) > 0.01:
#                    print("DANGER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "section", klm_hit_section[e][i], "sector", klm_hit_sector[e][i])
#                print("")

        klm1_phi_event = []
        klm1_the_event = []
        klm1_eng_event = []
        for i in klm_layer_1_indexes:
            for s0 in range(klm_hit_strip[e][i][0], klm_hit_last_strip[e][i][0] + 1):
                for s1 in range(klm_hit_strip[e][i][1], klm_hit_last_strip[e][i][1] + 1):
                    phithe = klm_hit_position(klm_hit_layer[e][i], klm_hit_sector[e][i], klm_hit_subdetector[e][i], klm_hit_section[e][i], s0, s1)
                    klm1_phi_event.append(phithe[0])
                    klm1_the_event.append(phithe[1])
                    klm1_eng_event.append(klm_hit_eng[e][i])

        m_klm1_phi_arrays.append(np.array(klm1_phi_event))
        m_klm1_the_arrays.append(np.array(klm1_the_event))
        m_klm1_eng_arrays.append(np.array(klm1_eng_event))
        klm1_n_hit_array[e] = len(klm1_phi_event)

        klm2_phi_event = []
        klm2_the_event = []
        klm2_eng_event = []
        for i in klm_layer_2_indexes:
            for s0 in range(klm_hit_strip[e][i][0], klm_hit_last_strip[e][i][0] + 1):
                for s1 in range(klm_hit_strip[e][i][1], klm_hit_last_strip[e][i][1] + 1):
                    phithe = klm_hit_position(klm_hit_layer[e][i], klm_hit_sector[e][i], klm_hit_subdetector[e][i], klm_hit_section[e][i], s0, s1)
                    klm2_phi_event.append(phithe[0])
                    klm2_the_event.append(phithe[1])
                    klm2_eng_event.append(klm_hit_eng[e][i])

        m_klm2_phi_arrays.append(np.array(klm2_phi_event))
        m_klm2_the_arrays.append(np.array(klm2_the_event))
        m_klm2_eng_arrays.append(np.array(klm2_eng_event))
        klm2_n_hit_array[e] = len(klm2_phi_event)

#        print("event info: n_hit:", klm1_n_hit_array[e])
#        print("phi array:", m_klm1_phi_arrays[e])
#        print("theta array:", m_klm1_the_arrays[e])
#        print("energy array:", m_klm1_eng_arrays[e])



        klm3_phi_event = []
        klm3_the_event = []
        klm3_eng_event = []
        for i in klm_layer_3_indexes:
            for s0 in range(klm_hit_strip[e][i][0], klm_hit_last_strip[e][i][0] + 1):
                for s1 in range(klm_hit_strip[e][i][1], klm_hit_last_strip[e][i][1] + 1):
                    phithe = klm_hit_position(klm_hit_layer[e][i], klm_hit_sector[e][i], klm_hit_subdetector[e][i], klm_hit_section[e][i], s0, s1)
                    klm3_phi_event.append(phithe[0])
                    klm3_the_event.append(phithe[1])
                    klm3_eng_event.append(klm_hit_eng[e][i])

        m_klm3_phi_arrays.append(np.array(klm3_phi_event))
        m_klm3_the_arrays.append(np.array(klm3_the_event))
        m_klm3_eng_arrays.append(np.array(klm3_eng_event))
        klm3_n_hit_array[e] = len(klm3_phi_event)

    for i in range(num_event):
        pass
#        print("Event " + str(i) + ":")
#        print(phi_RM_array[i])
#        print(sorted(m_phi_arrays[i])[len(m_phi_arrays[i])//2])
#        print("")
#        print(the_RM_array[i])
#        print(sorted(m_the_arrays[i])[len(m_the_arrays[i])//2])
#        print("")
#    for i in range(100):
#        print("phi_RM: ",  phi_RM_array[i])
#        print("the_RM: ", the_RM_array[i])
#        print("phi hits: ", m_phi_arrays[i])
#        print("theta hits: ", m_the_arrays[i])
#        print("")

    flag_cc_array = [1] * num_event
    flag_SB_array = [1] * num_event

    # for key, value in data.items():
    #     print(key, len(value))

    # flag_cc_array = data['flag_cc'] # flag_cc: used to set category_id in final json
    # flag_SB_array = data['flag_SB'] # flag_SB: used to set category_id in final json

    # p_RM_array = data['mcP'] # p_RM: Momentum? Or radius?
    # phi_RM_array = data['clusterPhi'] # phi_RM: phi angle of each event
    # the_RM_array = data['clusterTheta'] # the_RM: theta angle of each event

    # n_hit_array = data['clusterNHits'] # n_hit: Number of hits in each event
    # m_eng_arrays = data['m_eng'] # m_eng
    # m_phi_arrays = data['m_phi'] # m_phi
    # m_the_arrays = data['m_the'] # m_the
    # m_time_arrays = data['m_time'] # m_time:                                # The current value set is {0.0, 1.0, 2.0,..., 20.0}

    # num_event = len(n_hit_array)
    # print("num_event:", num_event)

    split_num = (num_event // split_size) + 1

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
        info = {"description": "Belle 2 Calorimeter Dataset",
                "url": "",
                "version": "0.01b",
                "year": 2025,
                "contributor": "Sean Frett",
                "date_created": "2025/02/10"}
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
        #print("i_begin: {}, i_end: {}".format(i_begin, i_end))              # 0~100000, 100000~200000, ...

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

            klm1_n_hit_i = int(klm1_n_hit_array[i])
            m_klm1_eng_array = np.array(m_klm1_eng_arrays[i]).astype(float)
            m_klm1_phi_array = np.array(m_klm1_phi_arrays[i]).astype(float)
            m_klm1_the_array = np.array(m_klm1_the_arrays[i]).astype(float)
            m_klm1_time_array = np.array([1] * klm1_n_hit_i).astype(int)

            klm2_n_hit_i = int(klm2_n_hit_array[i])
            m_klm2_eng_array = np.array(m_klm2_eng_arrays[i]).astype(float)
            m_klm2_phi_array = np.array(m_klm2_phi_arrays[i]).astype(float)
            m_klm2_the_array = np.array(m_klm2_the_arrays[i]).astype(float)
            m_klm2_time_array = np.array([1] * klm2_n_hit_i).astype(int)

            klm3_n_hit_i = int(klm3_n_hit_array[i])
            m_klm3_eng_array = np.array(m_klm3_eng_arrays[i]).astype(float)
            m_klm3_phi_array = np.array(m_klm3_phi_arrays[i]).astype(float)
            m_klm3_the_array = np.array(m_klm3_the_arrays[i]).astype(float)
            m_klm3_time_array = np.array([1] * klm3_n_hit_i).astype(int)

            category_id = int(flag_cc_i * 0.5 + flag_SB_i * 2.0 - 0.499)

            x_ctr_array, y_ctr_array, w_array, h_array, xmin_array, ymax_array = phithe_to_xywh_np(m_phi_array, m_the_array, width=width, height=height)
            xmax_array = xmin_array + w_array
            ymin_array = ymax_array - h_array
            xyxy_array = np.vstack([xmin_array, ymin_array, xmax_array, ymax_array]).T

            klm_x_ctr_array, klm_y_ctr_array, klm_w_array, klm_h_array, klm_xmin_array, klm_ymax_array = phithe_to_xywh_np(m_klm1_phi_array, m_klm1_the_array, width=width, height=height)
            klm_xmax_array = klm_xmin_array + klm_w_array
            klm_ymin_array = klm_ymax_array - klm_h_array
            klm1_xyxy_array = np.vstack([klm_xmin_array, klm_ymin_array, klm_xmax_array, klm_ymax_array]).T

            klm_x_ctr_array, klm_y_ctr_array, klm_w_array, klm_h_array, klm_xmin_array, klm_ymax_array = phithe_to_xywh_np(m_klm2_phi_array, m_klm2_the_array, width=width, height=height)
            klm_xmax_array = klm_xmin_array + klm_w_array
            klm_ymin_array = klm_ymax_array - klm_h_array
            klm2_xyxy_array = np.vstack([klm_xmin_array, klm_ymin_array, klm_xmax_array, klm_ymax_array]).T

            klm_x_ctr_array, klm_y_ctr_array, klm_w_array, klm_h_array, klm_xmin_array, klm_ymax_array = phithe_to_xywh_np(m_klm3_phi_array, m_klm3_the_array, width=width, height=height)
            klm_xmax_array = klm_xmin_array + klm_w_array
            klm_ymin_array = klm_ymax_array - klm_h_array
            klm3_xyxy_array = np.vstack([klm_xmin_array, klm_ymin_array, klm_xmax_array, klm_ymax_array]).T

            image_i += 1

            single_image = {}
            single_image['file_name'] = _filename + "_{:08}".format(image_i) + ".png"
            single_image['id'] = image_i
            single_image['width'] = width
            single_image['height'] = height
            single_image['ecl'] = {}
            single_image['ecl']['n_hit'] = n_hit_i
            single_image['ecl']['m_eng'] = m_eng_array.tolist()
            # single_image['m_phi'] = m_phi_array.tolist()
            # single_image['m_the'] = m_the_array.tolist()
            single_image['ecl']['xyxy'] = xyxy_array.astype(int).tolist()
            single_image['ecl']['m_time'] = m_time_array.tolist()

            single_image['klm1'] = {}
            single_image['klm1']['n_hit'] = klm1_n_hit_i
            single_image['klm1']['m_eng'] = m_klm1_eng_array.tolist()
            single_image['klm1']['xyxy'] = klm1_xyxy_array.astype(int).tolist()
            single_image['klm1']['m_time'] = m_klm1_time_array.tolist()

            single_image['klm2'] = {}
            single_image['klm2']['n_hit'] = klm2_n_hit_i
            single_image['klm2']['m_eng'] = m_klm2_eng_array.tolist()
            single_image['klm2']['xyxy'] = klm2_xyxy_array.astype(int).tolist()
            single_image['klm2']['m_time'] = m_klm2_time_array.tolist()

            single_image['klm3'] = {}
            single_image['klm3']['n_hit'] = klm3_n_hit_i
            single_image['klm3']['m_eng'] = m_klm3_eng_array.tolist()
            single_image['klm3']['xyxy'] = klm3_xyxy_array.astype(int).tolist()
            single_image['klm3']['m_time'] = m_klm3_time_array.tolist()

            data_dict['images'].append(single_image)

            if easy_scale > 0.0:
                x_ctr_float, y_ctr_float, w_int, h_int, _r5, _r6 = phithe_to_xywh_np(phi_RM_i, the_RM_i, width=width, height=height)
                #print("x_ctr_array:", x_ctr_array)
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
                
                #print("inbbox:", inbbox)
                #print("eng array[inbbox]:", m_eng_array[inbbox])
                #print(len(inbbox))
                #print(len(m_eng_array))

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
                # print(data_dict)
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
    parser.add_argument("--destroot", type = str, default = ".", help = "destroot")
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

