import os
import sys
import numpy as np
import copy
import csv
import math

# For generating Icosphere ...
import trimesh
from trimesh import creation

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import get_param as parameter
from apkit.apkit.doa import load_pts_horizontal

params = parameter.get_params()
_tdoa_saved_labels = params['saved_unique_classes_tdoa']
_tdoa_labels_dict = params['gt_dict']
_doa_labels = params['unique_classes_doa']

_pts_horizontal = load_pts_horizontal()
# _pts_3d = params['pts_3d']
nu = 3
_icosphere = trimesh.creation.icosphere(nu)
_pts_3d = np.array(_icosphere.vertices)

def read_CSV(file_name):
    total_labels = copy.deepcopy(_tdoa_labels_dict)

    # Read CSV file
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        tmp_labels = list(reader)

        for tmp_label in tmp_labels:

            for key in _tdoa_saved_labels:
                if tmp_label[1] == _tdoa_saved_labels[key]:
                    if key == 'femalescream' or key == 'malescream':
                        labels_info = total_labels['scream'][0]
                        labels_xyz = total_labels['scream'][1]
                        labels_info.append(list(map(int, tmp_label[:3])))
                        labels_xyz.append(list(map(float, tmp_label[3:])))
                    elif key == 'femalespeech' or key == 'malespeech':
                        labels_info = total_labels['speech'][0]
                        labels_xyz = total_labels['speech'][1]
                        labels_info.append(list(map(int, tmp_label[:3])))
                        labels_xyz.append(list(map(float, tmp_label[3:])))
                    else:
                        labels_info = total_labels[key][0]
                        labels_xyz = total_labels[key][1]
                        labels_info.append(list(map(int, tmp_label[:3])))
                        labels_xyz.append(list(map(float, tmp_label[3:])))

    return total_labels  # Return Dictionary of labels

def compute_delay(mic_pairs_pos, doa, c=340, fs=24000):
    """
    :param mic_pairs_pos: microphone pair's positions, (M, 2, 3) array,
                          M is the unique number of microphone pair
    :param doa: Normalized direction of arrival, (3,) array or (N, 3) array
                N is the number of sources
    :param c: speed of sound (m/s)

    :return:
    """
    mic_pairs_pos = np.array(mic_pairs_pos)
    doa = np.array(doa)

    r_mic_pairs_pos = mic_pairs_pos[:, 1, :] - mic_pairs_pos[:, 0, :]  # (M, 3) array

    if doa.ndim == 1:
        diff = -np.einsum('ij,j->i', r_mic_pairs_pos, doa) / c
    else:
        assert doa.ndim == 2
        diff = -np.einsum('ij,kj->ki', r_mic_pairs_pos, doa) / c

    return diff * fs

def extract_diff_TDOA(total_labels, mic_center_pos, mic_pair_pos, num_half_delays, sampling_rate=24000):
    labels_diff_pair = dict()

    for key in total_labels:
        gts_xyz = total_labels[key][1]
        if len(gts_xyz) > 0:
            # normalization
            doa_vec = np.array(gts_xyz) - np.repeat(np.expand_dims(mic_center_pos, axis=0), repeats=len(gts_xyz),
                                                    axis=0)
            norm_doa_vec = np.linalg.norm(doa_vec, axis=-1)
            doa_vec = doa_vec / np.repeat(np.expand_dims(norm_doa_vec, axis=1), repeats=3, axis=1)

            diff = compute_delay(mic_pair_pos, doa_vec, fs=sampling_rate)
            labels_diff_pair[key] = np.around(diff).astype(dtype=int) + num_half_delays
        else:
            labels_diff_pair[key] = np.empty([0, 0])

    return labels_diff_pair

def extract_TDOA_labels(total_labels, labels_diff_pair, nframe, npairs, num_half_delays):
    ndelays = num_half_delays * 2 + 1
    gts_pair = np.zeros((len(total_labels), nframe, npairs, ndelays))
    gts_pair_xyz = np.zeros((len(total_labels), nframe, npairs, 1 + 3))

    for kix, key in enumerate(total_labels):
        gts_info = total_labels[key][0]
        diff_int = labels_diff_pair[key]
        diff_int_shape = diff_int.shape

        for idx in range(diff_int_shape[0]):
            frame_idx = gts_info[idx][0]
            for idxp in range(diff_int_shape[1]):
                ### --> Batch * Time bins (10) * Delays (51)
                gts_pair[kix, frame_idx, idxp, diff_int[idx, idxp]] = 1

                ### --> Batch * Time bins (10) * [Num delay, delays] (1+3*3)
                start_idx = int(gts_pair_xyz[kix, frame_idx, idxp, 0])
                gts_pair_xyz[kix, frame_idx, idxp, start_idx + 1] = diff_int[idx, idxp]
                gts_pair_xyz[kix, frame_idx, idxp, 0] += 1

    return gts_pair, gts_pair_xyz

def extract_DOA_labels(total_labels, nframe, mic_center_pos, sigma_sq=0.005, out_dim=3, dataset_flag=parameter.SSLR_DATASET):
    ### Only speech
    if dataset_flag == parameter.SSLR_DATASET:
        numpy_gts = np.zeros((nframe, 3 * 3 + 1))
        encoded_like_gt_ovr_fr = np.zeros((nframe, 360))

        for key in total_labels:
            if key == 'speech':
                gts_info = total_labels[key][0]
                gts_xyz = total_labels[key][1]
                for gt_info, gt_xyz in zip(gts_info, gts_xyz):
                    fix = int(gt_info[0])

                    src_pos = np.array(gt_xyz)
                    doa_vec = src_pos - mic_center_pos
                    doa_vec = doa_vec / np.linalg.norm(doa_vec)
                    idx_doa = int(numpy_gts[fix, 0])  # Get the number of GTs
                    numpy_gts[fix, idx_doa * 3 + 1:(idx_doa + 1) * 3 + 1] = doa_vec
                    numpy_gts[fix, 0] += 1  # Increase the number of GTs

                    ### Compute angular distance
                    doa_vec_horizontal = doa_vec[:2]
                    doa_vec_horizontal = np.tile(doa_vec_horizontal, (360, 1))
                    pts_horizontal = _pts_horizontal[:, :2]
                    denom = np.linalg.norm(_pts_horizontal, axis=1) * np.linalg.norm(doa_vec_horizontal,
                                                                                          axis=1)
                    denom[denom < 1e-16] = math.pi
                    sim = np.sum(pts_horizontal * doa_vec_horizontal, axis=1) / denom
                    sim[sim > 1.0] = 1.0
                    sim[sim < -1.0] = -1.0
                    _angle_dist = np.arccos(sim)

                    # Compute Gaussian-like function
                    _gauss_like_func = np.exp(-_angle_dist * _angle_dist / sigma_sq)

                    # Check max value
                    encoded_like_gt_ovr_fr[fix, encoded_like_gt_ovr_fr[fix, :] < _gauss_like_func] = \
                        _gauss_like_func[
                            encoded_like_gt_ovr_fr[fix, :] < _gauss_like_func]

        return numpy_gts, encoded_like_gt_ovr_fr

    ### Speech + other labels
    elif dataset_flag == parameter.DCASE2021_DATASET:
        num_labels = len(_doa_labels)
        pts_xyz = None
        if out_dim == 2:
            pts_xyz = _pts_horizontal
        elif out_dim == 3:
            pts_xyz = _pts_3d
        num_pts = len(pts_xyz)

        numpy_gts = np.zeros((nframe, out_dim * num_labels))
        encoded_like_gt_ovr_fr = np.zeros((nframe, num_pts, num_labels))

        for key in _doa_labels:  # For all keys of sound events
            class_idx = int(_doa_labels[key])
            gts_info = total_labels[key][0]
            gts_xyz = total_labels[key][1]
            for gt_info, gt_xyz in zip(gts_info, gts_xyz):
                fix = int(gt_info[0])

                # src_pos = np.array(list(map(float, gt_xyz)), dtype=np.float)
                src_pos = np.array(gt_xyz)
                doa_vec = src_pos - mic_center_pos
                doa_vec = doa_vec / np.linalg.norm(doa_vec)

                # numpy_gts[fix, class_idx * 3 : (class_idx+1) * 3] = doa_vec
                numpy_gts[fix, class_idx] = doa_vec[0]
                numpy_gts[fix, class_idx + num_labels] = doa_vec[1]
                if out_dim > 2:
                    numpy_gts[fix, class_idx + num_labels * 2] = doa_vec[2]

                ### Compute Gaussian-like function for 2D or 3D
                # pts_xyz = None
                doa_vec_xyz = None
                if out_dim == 2:
                    # pts_xyz = self._pts_horizontal
                    doa_vec_xyz = np.zeros(3)
                    doa_vec_xyz[:2] = doa_vec[:2]
                elif out_dim == 3:
                    # pts_xyz = self._pts_3d
                    doa_vec_xyz = doa_vec

                doa_vec_xyz = np.tile(doa_vec_xyz, (num_pts, 1))

                denom = np.linalg.norm(pts_xyz, axis=1) * np.linalg.norm(doa_vec_xyz, axis=1)
                denom[denom < 1e-16] = math.pi
                sim = np.sum(pts_xyz * doa_vec_xyz, axis=1) / denom
                sim[sim > 1.0] = 1.0
                sim[sim < -1.0] = -1.0
                _angle_dist = np.arccos(sim)

                # Compute Gaussian-like function
                _gauss_like_func = np.exp(-_angle_dist * _angle_dist / sigma_sq)

                # Check max value
                encoded_like_gt_ovr_fr[fix, encoded_like_gt_ovr_fr[fix, :, class_idx] < _gauss_like_func, class_idx] = \
                    _gauss_like_func[encoded_like_gt_ovr_fr[fix, :, class_idx] < _gauss_like_func]
        return numpy_gts, encoded_like_gt_ovr_fr
    else:
        print("[Error] The flag choosing datasets is not correct !!! ")