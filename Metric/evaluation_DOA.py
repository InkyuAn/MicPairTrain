import math
import sys
import os
import numpy as np

from scipy.signal import find_peaks
from scipy.spatial import distance

from Metric.seld_metric import SELD_evaluation_metrics

eps = np.finfo(np.float64).eps

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from apkit.apkit.doa import azimuth_distance, load_pts_horizontal

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

def get_SELD_Results(_pred_dict, _gt_dict, _nb_classes, _doa_thresh):

        # total_fr_num = len(_gt_dict)

        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=_nb_classes, doa_threshold=_doa_thresh)
        # for pred_cnt in range(total_fr_num):
        #     # Load predicted output format file
        #     pred_dict = _pred_dict[pred_cnt]
        #     gt_dict = _gt_dict[pred_cnt]
        #
        #     # Calculated scores
        #     eval.update_seld_scores(pred_dict, gt_dict)

        # Load predicted output format file
        pred_dict = _pred_dict
        gt_dict = _gt_dict

        # Calculated scores
        eval.update_seld_scores(pred_dict, gt_dict)

        # Overall SED and DOA scores
        ER, F, LE, LR = eval.compute_seld_scores()
        seld_scr = SELD_evaluation_metrics.early_stopping_metric([ER, F], [LE, LR])

        return ER, F, LE, LR, seld_scr

def segment_labels(_pred_dict, _max_frames, _nb_label_frames_1s):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(_nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, _nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // _nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+_nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

def regression_label_format_to_output_format(_sed_labels, _doa_labels, _nb_classes, out_dim=3):
    """
    Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

    :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
    :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
    :return: _output_dict: returns a dict containing dcase output format
    """


    _x, _y, _z = None, None, None
    if out_dim == 2:
        _x = _doa_labels[:, :_nb_classes]
        _y = _doa_labels[:, _nb_classes:]
    elif out_dim == 3:
        _x = _doa_labels[:, :_nb_classes]
        _y = _doa_labels[:, _nb_classes:2*_nb_classes]
        _z = _doa_labels[:, 2*_nb_classes:]

    _output_dict = {}
    for _frame_ind in range(_sed_labels.shape[0]):
        _tmp_ind = np.where(_sed_labels[_frame_ind, :])
        if len(_tmp_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_ind[0]:
                if out_dim == 2:
                    _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class]])
                elif out_dim == 3:
                    _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
    return _output_dict

def get_accdoa_labels(accdoa_in, nb_classes, out_dim=3, threshold=0.5):
    if out_dim == 3:
        x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
        sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > threshold

        return sed, accdoa_in
    elif out_dim == 2:
        x, y = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes]
        sed = np.sqrt(x ** 2 + y ** 2) > threshold

        return sed, accdoa_in

def get_accdoa_vectors(accdoa_in, nb_classes, out_dim=3):
    if out_dim == 3:
        x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
        sed_score = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return sed_score, x, y, z
    elif out_dim == 2:
        x, y = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes]
        sed_score = np.sqrt(x ** 2 + y ** 2)
        return sed_score, x, y


class evaluation_SELD:
    def __init__(self, doa_threshold=5, pred_threshold=0.2):
    # def __init__(self, doa_threshold=5, pred_threshold=0.15):
        self._pts_horizontal = load_pts_horizontal()

        # self._num_fr = num_frame
        self._num_gt_exceed = 0

        # Variables for speech detection(speech)
        self._TP = 0
        self._FP = 0
        self._FN = 0

        self._S = 0
        self._D = 0
        self._I = 0
        self._Nref = 0

        self._spatial_T = doa_threshold * np.pi / 180
        self._pred_T = pred_threshold

        # variables for DoAs
        self._total_DE = 0
        self._total_DE_Srcs = [0, 0]

        self._DE_TP = 0
        self._DE_TP_Srcs = [0, 0]
        self._DE_FP = 0
        self._DE_FN = 0

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # Location-sensitive detection performance
        ER = (self._S + self._D + self._I) / float(self._Nref + eps)
        F = self._TP / (eps + self._TP + 0.5 * (self._FP + self._FN))

        # Class-sensitive localization performance
        # LE = self._total_DE / float(self._DE_TP + eps) if self._DE_TP else 180 # When the total number of prediction is zero
        LE = self._total_DE / float(
            self._DE_TP + eps) if self._DE_TP else np.pi  # When the total number of prediction is zero
        LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)
        # print("Debugging, DE_TP: ", self._DE_TP, ", DE_TP_Srcs[0]: ", self._DE_TP_Srcs[0], ", DE_TP_Srcs[1]: ", self._DE_TP_Srcs[1], ", ... Num. gts over 3: ", self._num_gt_exceed)
        # print('S {}, D {}, I {}, Nref {}, TP {}, FP {}, FN {}, DE_TP {}, DE_FN {}, totalDE {}'.format(self._S, self._D, self._I, self._Nref, self._TP, self._FP, self._FN, self._DE_TP, self._DE_FN, self._total_DE))
        # return ER, F, LE, LR
        return ER, F, LE * 180 / math.pi, LR

    def early_stopping_metric(self, sed_error, doa_error):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            sed_error[0],
            1 - sed_error[1],
            doa_error[0] / np.pi,
            1 - doa_error[1]]
        )
        return seld_metric

    def decoding_pred(self, pred):
        peak_list = []
        neighbor_angle_dists = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
        # threshold_pred = 0.125
        # threshold_pred = 0.2

        for idx_ang in range(360):
            if pred[idx_ang] > self._pred_T:
                is_peak = True
                for angle_dist in neighbor_angle_dists:
                    idx_neig_ang = angle_dist + idx_ang
                    if idx_neig_ang < 0:
                        idx_neig_ang = idx_neig_ang + 360
                    elif idx_neig_ang >= 360:
                        idx_neig_ang = idx_neig_ang - 360
                    if pred[idx_ang] < pred[idx_neig_ang]:
                        is_peak = False
                if is_peak:
                    peak_list.append((idx_ang, pred[idx_ang]))

        peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
        return peak_list

    # def evaluation_MAE(self, nFrame, num_of_gt, gt_doas, pred, threshold=0.2):
    def update_seld_scores(self, num_fr, num_of_gts, gt_doas, pred):



        for idx_fr in range(num_fr):
            # print("Start frame: ", idx_fr)

            loc_FN, loc_FP = 0, 0

            # Compute peaks
            peak_indice = find_peaks(pred[idx_fr])

            peak_list = np.array([peak_indice[0], pred[idx_fr, peak_indice[0]]]).transpose().tolist()
            peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
            peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[1]]

            ###
            # peak_list2 = self.decoding_pred(pred[idx_fr])
            num_of_gt_ref = int(num_of_gts[idx_fr])
            num_of_gt_pred = len(peak_list)

            if num_of_gt_ref > 0:
                self._Nref += num_of_gt_ref

            if num_of_gt_pred == num_of_gt_ref:
                self._DE_TP += num_of_gt_ref
            elif num_of_gt_pred < num_of_gt_ref:
                self._DE_FN += (num_of_gt_ref - num_of_gt_pred)
                self._DE_TP += num_of_gt_pred
            elif num_of_gt_pred > num_of_gt_ref:
                self._DE_FP += (num_of_gt_pred - num_of_gt_ref)
                self._DE_TP += num_of_gt_ref

            ######################################################################
            # print("  Debugging point, 0")
            ######################################################################


            ### Compute Distance
            # for idx_gt_pred in range(num_of_gt_pred):
            for idx_gt_pred in range(min(num_of_gt_pred, num_of_gt_ref)):
                tmp_pred_doa_idx = int(peak_list[idx_gt_pred][0])

                ######################################################################
                # print("  Debugging point, Compute dist, 0")
                ######################################################################

                min_angle_dist = 100
                for idx_gt_ref in range(num_of_gt_ref):
                    # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
                    tmp_gt_doa = gt_doas[idx_fr, idx_gt_ref * 3:(idx_gt_ref + 1) * 3]

                    # print("  Debugging point, before compute_dist")
                    # print("  Debugging point, pts_hor...: ", tmp_pred_doa_idx)
                    # print("  Debugging point, pts_hor...: ", self._pts_horizontal[tmp_pred_doa_idx].shape, "self._pts_horizontal: ", self._pts_horizontal[tmp_pred_doa_idx])
                    # print("  Debugging point, pts_hor...", tmp_gt_doa.shape, ", tmp_gt: ", tmp_gt_doa)
                    # print("  Debugging point, azimuth_distance...", azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))
                    # print("  Check point, ")
                    angle_dist = abs(azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))
                    # print("  Debugging point, after compute_dist")
                    # delay_diff = abs(tmp_gt_doa - tmp_pred_delay)

                    if min_angle_dist > angle_dist:
                        min_angle_dist = angle_dist
                        # self._total_DE += delay_diff
                        # self._total_DE_Srcs[0] += delay_diff

                ######################################################################
                # print("  Debugging point, Compute dist, 1")
                ######################################################################

                if min_angle_dist < 99:  # Check exception condition
                    self._total_DE += min_angle_dist
                    if num_of_gt_ref == 1:
                        self._total_DE_Srcs[0] += min_angle_dist
                        self._DE_TP_Srcs[0] += 1
                    elif num_of_gt_ref == 2:
                        self._total_DE_Srcs[1] += min_angle_dist
                        self._DE_TP_Srcs[1] += 1
                else:
                    print("There is some issues!!! Check it ")

                ######################################################################
                # print("  Debugging point, Compute dist, 2")
                ######################################################################

                # TP_FN_array[i] == 1: TP, TP_FN_array[i] == 0: FN
                TP_FN_array = np.zeros(num_of_gt_ref)
                # FP_array[i] == 1: FP, TP_FN_array[i] == 0: TP or TPs
                FP_array = np.zeros(num_of_gt_pred)
                for idx_gt_ref in range(num_of_gt_ref):
                    # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
                    tmp_gt_doa = gt_doas[idx_fr, idx_gt_ref * 3:(idx_gt_ref + 1) * 3]

                    for idx_gt_pred in range(num_of_gt_pred):
                        # tmp_pred_delay = peak_list[idx_gt_pred][0]
                        tmp_pred_doa_idx = int(peak_list[idx_gt_pred][0])

                        # delay_diff = abs(tmp_gt_delay - tmp_pred_delay)
                        # print("  Debugging point, before compute_dist")
                        # print("  Debugging point, pts_hor...: ", tmp_pred_doa_idx)
                        # print("  Debugging point, pts_hor...: ", self._pts_horizontal[tmp_pred_doa_idx].shape, "self._pts_horizontal: ", self._pts_horizontal[tmp_pred_doa_idx])
                        # print("  Debugging point, pts_hor...", tmp_gt_doa.shape, ", tmp_gt: ", tmp_gt_doa)
                        # print("  Debugging point, azimuth_distance...",
                        #       azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))
                        angle_dist = abs(azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))
                        # print("  Debugging point, after compute_dist")

                        if angle_dist < self._spatial_T:
                            # TP
                            TP_FN_array[idx_gt_ref] = 1
                        else:
                            # FP
                            FP_array[idx_gt_pred] = 1

                ######################################################################
                # print("  Debugging point, Compute dist, 3")
                ######################################################################

                ### Compute SED score
                for idx_gt_ref in range(num_of_gt_ref):
                    if TP_FN_array[idx_gt_ref] == 1:  # TP
                        self._TP += 1
                    else:  # FN
                        self._FN += 1
                        loc_FN += 1

                for idx_gt_pred in range(num_of_gt_pred):
                    if FP_array[idx_gt_pred] == 1:  # FP
                        self._FP += 1
                        loc_FP += 1

                ######################################################################
                # print("  Debugging point, Compute dist, 4")
                ######################################################################

                self._S += np.minimum(loc_FP, loc_FN)
                self._D += np.maximum(0, loc_FN - loc_FP)
                self._I += np.maximum(0, loc_FP - loc_FN)

        ## For all frames
        # for idx_fr in range(num_fr):
        #
        #     loc_FN, loc_FP = 0, 0
        #
        #     peak_list = self.decoding_pred(pred[idx_fr])
        #     num_of_gt_ref = int(num_of_gts[idx_fr])
        #     num_of_gt_pred = len(peak_list)
        #
        #     # num_gt = num_gt + num_of_gt
        #     # num_all_pred = num_all_pred + len(peak_list)
        #
        #     if num_of_gt_ref > 0:
        #         self._Nref += num_of_gt_ref
        #
        #     if num_of_gt_pred == num_of_gt_ref:
        #         self._DE_TP += num_of_gt_ref
        #     elif num_of_gt_pred < num_of_gt_ref:
        #         self._DE_FN += (num_of_gt_ref-num_of_gt_pred)
        #         self._DE_TP += num_of_gt_pred
        #     elif num_of_gt_pred > num_of_gt_ref:
        #         self._DE_FP += (num_of_gt_pred-num_of_gt_ref)
        #         self._DE_TP += num_of_gt_ref
        #
        #
        #
        #     if num_of_gt_ref == 1:
        #         DoA_vec = gt_doas[idx_fr, 0 * 3:(0 + 1) * 3]
        #
        #         if num_of_gt_pred >= 1:
        #             angle_dist = abs(azimuth_distance(self._pts_horizontal[peak_list[0][0]], DoA_vec))
        #
        #             # list_MAE_angle.append(angle_dist)
        #             # list_MAE_angle_N_1.append(angle_dist)
        #             self._total_DE += angle_dist
        #             self._total_DE_Srcs[0] += angle_dist
        #             # self._DE_TP += 1
        #             self._DE_TP_Srcs[0] += 1
        #
        #             if angle_dist < self._spatial_T:
        #                 # num_correct_detection = num_correct_detection + 1
        #                 self._TP += 1
        #             else:
        #                 loc_FP += 1
        #                 self._FP += 1
        #         else:
        #             loc_FN += 1
        #             self._FN += 1
        #             # self._DE_FN += 1
        #     elif num_of_gt_ref == 2:
        #         DoA_vec_0 = gt_doas[idx_fr, 0 * 3:(0 + 1) * 3]
        #         DoA_vec_1 = gt_doas[idx_fr, 1 * 3:(1 + 1) * 3]
        #
        #         if num_of_gt_pred >= 2:
        #             # num_correct_detection = num_correct_detection + 2
        #             angle_dist_0_0 = abs(azimuth_distance(self._pts_horizontal[peak_list[0][0]], DoA_vec_0))
        #             angle_dist_1_1 = abs(azimuth_distance(self._pts_horizontal[peak_list[1][0]], DoA_vec_1))
        #
        #             angle_dist_0_1 = abs(azimuth_distance(self._pts_horizontal[peak_list[0][0]], DoA_vec_1))
        #             angle_dist_1_0 = abs(azimuth_distance(self._pts_horizontal[peak_list[1][0]], DoA_vec_0))
        #
        #             if angle_dist_0_0 + angle_dist_1_1 < angle_dist_0_1 + angle_dist_1_0:
        #                 # list_MAE_angle.append(angle_dist_0_0)
        #                 # list_MAE_angle.append(angle_dist_1_1)
        #                 # list_MAE_angle_N_2.append(angle_dist_0_0)
        #                 # list_MAE_angle_N_2.append(angle_dist_1_1)
        #                 self._total_DE += (angle_dist_0_0 + angle_dist_1_1)
        #                 self._total_DE_Srcs[1] += (angle_dist_0_0 + angle_dist_1_1)
        #                 # self._DE_TP += 2
        #                 self._DE_TP_Srcs[1] += 2
        #
        #                 if angle_dist_0_0 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #
        #                 if angle_dist_1_1 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #             else:
        #                 # list_MAE_angle.append(angle_dist_0_1)
        #                 # list_MAE_angle.append(angle_dist_1_0)
        #                 # list_MAE_angle_N_2.append(angle_dist_0_1)
        #                 # list_MAE_angle_N_2.append(angle_dist_1_0)
        #                 self._total_DE += (angle_dist_0_1 + angle_dist_1_0)
        #                 self._total_DE_Srcs[1] += (angle_dist_0_1 + angle_dist_1_0)
        #                 # self._DE_TP += 2
        #                 self._DE_TP_Srcs[1] += 2
        #
        #                 if angle_dist_0_1 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #                 if angle_dist_1_0 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #         elif num_of_gt_pred == 1:
        #             # num_correct_detection = num_correct_detection + 1
        #             angle_dist_0_0 = abs(azimuth_distance(self._pts_horizontal[peak_list[0][0]], DoA_vec_0))
        #             angle_dist_0_1 = abs(azimuth_distance(self._pts_horizontal[peak_list[0][0]], DoA_vec_1))
        #
        #             if angle_dist_0_0 < angle_dist_0_1:
        #                 # list_MAE_angle.append(angle_dist_0_0)
        #                 # list_MAE_angle_N_2.append(angle_dist_0_0)
        #                 self._total_DE += angle_dist_0_0
        #                 self._total_DE_Srcs[1] += angle_dist_0_0
        #                 # self._DE_TP += 1
        #                 self._DE_TP_Srcs[1] += 1
        #                 if angle_dist_0_0 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #             else:
        #                 # list_MAE_angle.append(angle_dist_0_1)
        #                 # list_MAE_angle_N_2.append(angle_dist_0_1)
        #                 self._total_DE += angle_dist_0_1
        #                 self._total_DE_Srcs[1] += angle_dist_0_1
        #                 # self._DE_TP += 1
        #                 self._DE_TP_Srcs[1] += 1
        #
        #                 if angle_dist_0_1 < self._spatial_T:
        #                     # num_correct_detection = num_correct_detection + 1
        #                     self._TP += 1
        #                 else:
        #                     loc_FP += 1
        #                     self._FP += 1
        #             loc_FN += 1
        #             self._FN += 1
        #             # self._DE_FN += 1
        #         else:
        #             loc_FN += 2
        #             self._FN += 2
        #             # self._DE_FN += 2
        #     elif num_of_gt_ref >= 3:
        #         self._num_gt_exceed += 1
        #
        #     self._S += np.minimum(loc_FP, loc_FN)
        #     self._D += np.maximum(0, loc_FN - loc_FP)
        #     self._I += np.maximum(0, loc_FP - loc_FN)
        # return list_MAE_angle, list_MAE_angle_N_1, list_MAE_angle_N_2, num_correct_detection, num_all_pred, num_gt



class evaluation_SSLR:
    def __init__(self, doa_threshold=5, pred_threshold=0.45):
    # def __init__(self, doa_threshold=5, pred_threshold=0.2):
        self._pts_horizontal = load_pts_horizontal()

        # self._num_fr = num_frame
        self._num_gt_exceed = 0

        self._num_all_pred = 0
        self._num_all_gts = 0
        self._num_all_peaks = 0
        self._num_correct_pred = 0

        self._TP = 0
        # Variables for speech detection(speech)

        self._spatial_T = doa_threshold * np.pi / 180
        self._pred_T = pred_threshold

        # variables for DoAs
        self._total_DE = 0

        # Test for each number of sources
        self._total_DE_Srcs = [0, 0]
        self._num_correct_pred_Srcs = [0, 0]
        self._num_all_peaks_Srcs = [0, 0]

        self._num_all_pred_Srcs = [0, 0]
        self._num_all_gts_Srcs = [0, 0]
        self._TP_Srcs = [0, 0]

        #self._DE_TP_Srcs = [0, 0]

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # MAE = self._total_DE / self._num_all_peaks
        # ACC = self._num_correct_pred / self._num_all_peaks
        MAE = math.pi / 2. if self._num_all_peaks <= 0. else self._total_DE / self._num_all_peaks
        ACC = 0. if self._num_all_peaks <= 0. else self._num_correct_pred / self._num_all_peaks

        Precision = 0. if self._num_all_pred <= 0. else self._TP / self._num_all_pred
        Recall = 0. if self._num_all_gts <= 0. else self._TP / self._num_all_gts

        return MAE * 180 / math.pi, ACC, Precision, Recall

    def compute_score_each_src(self, num_of_src=1):
        idx_src = num_of_src - 1
        # MAE = self._total_DE_Srcs[idx_src] / self._num_all_peaks_Srcs[idx_src]
        # ACC = self._num_correct_pred_Srcs[idx_src] / self._num_all_peaks_Srcs[idx_src]
        MAE = math.pi / 2. if self._num_all_peaks_Srcs[idx_src] <= 0. else self._total_DE_Srcs[idx_src] / \
                                                                           self._num_all_peaks_Srcs[idx_src]
        ACC = 0. if self._num_all_peaks_Srcs[idx_src] <= 0 else self._num_correct_pred_Srcs[idx_src] / \
                                                                self._num_all_peaks_Srcs[idx_src]
        return MAE * 180 / math.pi, ACC

    def compute_prec_rec_curve_each_src(self, num_of_src=1):
        idx_src = num_of_src - 1
        # Precision = self._TP_Srcs[idx_src] / self._num_all_pred_Srcs[idx_src]
        # Recall = self._TP_Srcs[idx_src] / self._num_all_gts_Srcs[idx_src]
        Precision = 0. if self._num_all_pred_Srcs[idx_src] <= 0. else self._TP_Srcs[idx_src] / self._num_all_pred_Srcs[
            idx_src]
        Recall = 0. if self._num_all_pred_Srcs[idx_src] <= 0. else self._TP_Srcs[idx_src] / self._num_all_gts_Srcs[
            idx_src]

        return Precision, Recall

    def early_stopping_metric(self, sed_error, doa_error):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            sed_error[0],
            1 - sed_error[1],
            doa_error[0] / np.pi,
            1 - doa_error[1]]
        )
        return seld_metric


    # def evaluation_MAE(self, nFrame, num_of_gt, gt_doas, pred, threshold=0.2):
    def update_seld_scores(self, num_fr, num_of_gts, gt_doas, pred):

        for idx_fr in range(num_fr):
            # Compute peaks
            peak_indice = find_peaks(pred[idx_fr])

            peak_list = np.array([peak_indice[0], pred[idx_fr, peak_indice[0]]]).transpose().tolist()
            peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
            peak_list_th = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[1]]

            num_of_gt_ref = int(num_of_gts[idx_fr])
            num_of_gt_pred_th = len(peak_list_th)

            #########
            ### Unknown the number of sources (Precision & Recall)
            ### Denoms of Precision and recall
            self._num_all_gts += num_of_gt_ref
            self._num_all_pred += num_of_gt_pred_th

            if num_of_gt_ref == 1:
                self._num_all_gts_Srcs[0] += num_of_gt_ref
                self._num_all_pred_Srcs[0] += num_of_gt_pred_th
            elif num_of_gt_ref == 2:
                self._num_all_gts_Srcs[1] += num_of_gt_ref
                self._num_all_pred_Srcs[1] += num_of_gt_pred_th

            ### Compute Distance
            # for idx_gt_pred in range(num_of_gt_pred):
            for idx_gt_pred in range(min(num_of_gt_pred_th, num_of_gt_ref)):
                tmp_pred_doa_value = peak_list_th[idx_gt_pred][1]
                tmp_pred_doa_idx = int(peak_list_th[idx_gt_pred][0])

                min_angle_dist = 100
                for idx_gt_ref in range(num_of_gt_ref):
                    # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
                    tmp_gt_doa = gt_doas[idx_fr, idx_gt_ref * 3:(idx_gt_ref + 1) * 3]

                    angle_dist = abs(azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))

                    if min_angle_dist > angle_dist:
                        min_angle_dist = angle_dist

                if min_angle_dist < 99:  # Check exception condition
                    if min_angle_dist < self._spatial_T:
                        ### Compute Precision and recall
                        self._TP += 1

                        if num_of_gt_ref == 1:
                            self._TP_Srcs[0] += 1
                        elif num_of_gt_ref == 2:
                            self._TP_Srcs[1] += 1

                else:
                    print("There is some issues!!! Check it ")

            #########
            ### Known the number of sources (Precision & Recall)
            ### Denoms of Precision and recall
            for idx_gt_pred in range(num_of_gt_ref):
            # for idx_gt_pred in range(min(num_of_gt_pred_th, num_of_gt_ref)):
                ### IK 2022 07 12, added 'if len(peak_list) > 0:' ...
                if len(peak_list) > 0:

                    tmp_pred_doa_value = peak_list[idx_gt_pred][1]
                    tmp_pred_doa_idx = int(peak_list[idx_gt_pred][0])

                    min_angle_dist = 100
                    for idx_gt_ref in range(num_of_gt_ref):
                        # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
                        tmp_gt_doa = gt_doas[idx_fr, idx_gt_ref * 3:(idx_gt_ref + 1) * 3]

                        angle_dist = abs(azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))

                        if min_angle_dist > angle_dist:
                            min_angle_dist = angle_dist

                    if min_angle_dist < 99:  # Check exception condition
                        self._total_DE += min_angle_dist
                        self._num_all_peaks += 1
                        if num_of_gt_ref == 1:
                            self._total_DE_Srcs[0] += min_angle_dist
                            self._num_all_peaks_Srcs[0] += 1
                            # self._DE_TP_Srcs[0] += 1
                        elif num_of_gt_ref == 2:
                            self._total_DE_Srcs[1] += min_angle_dist
                            self._num_all_peaks_Srcs[1] += 1
                            # self._DE_TP_Srcs[1] += 1

                        if min_angle_dist < self._spatial_T:
                            self._num_correct_pred += 1

                            if num_of_gt_ref == 1:
                                self._num_correct_pred_Srcs[0] += 1
                            elif num_of_gt_ref == 2:
                                self._num_correct_pred_Srcs[1] += 1

                    else:
                        print("There is some issues!!! Check it ")

            # # Compute peaks
            # peak_indice = find_peaks(pred[idx_fr])
            #
            # peak_list = np.array([peak_indice[0], pred[idx_fr, peak_indice[0]]]).transpose().tolist()
            # peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
            # peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[1]]
            #
            # num_of_gt_ref = int(num_of_gts[idx_fr])
            # num_of_gt_pred = len(peak_list)
            #
            # ### Precision and recall
            # # self._num_all_pred += len(peak_list_thr)    # Denom of precision
            # self._num_all_gts += num_of_gt_ref    # Denom of recall
            # # self._num_all_peaks += min(num_of_gt_pred, num_of_gt_ref) # Denom of ACC
            # self._num_all_pred += num_of_gt_pred
            #
            # '''
            # self._total_DE_Srcs = [0, 0]
            # self._num_correct_pred_Srcs = [0, 0]
            # self._num_all_peaks_Srcs = [0, 0]
            # '''
            #
            # ### Compute Distance
            # # for idx_gt_pred in range(num_of_gt_pred):
            # for idx_gt_pred in range(min(num_of_gt_pred, num_of_gt_ref)):
            #     tmp_pred_doa_value = peak_list[idx_gt_pred][1]
            #     tmp_pred_doa_idx = int(peak_list[idx_gt_pred][0])
            #
            #     min_angle_dist = 100
            #     for idx_gt_ref in range(num_of_gt_ref):
            #         # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
            #         tmp_gt_doa = gt_doas[idx_fr, idx_gt_ref * 3:(idx_gt_ref + 1) * 3]
            #
            #         angle_dist = abs(azimuth_distance(self._pts_horizontal[tmp_pred_doa_idx], tmp_gt_doa))
            #
            #         if min_angle_dist > angle_dist:
            #             min_angle_dist = angle_dist
            #
            #     if min_angle_dist < 99:  # Check exception condition
            #         self._total_DE += min_angle_dist
            #         self._num_all_peaks += 1
            #         if num_of_gt_ref == 1:
            #             self._total_DE_Srcs[0] += min_angle_dist
            #             self._num_all_peaks_Srcs[0] += 1
            #             # self._DE_TP_Srcs[0] += 1
            #         elif num_of_gt_ref == 2:
            #             self._total_DE_Srcs[1] += min_angle_dist
            #             self._num_all_peaks_Srcs[1] += 1
            #             # self._DE_TP_Srcs[1] += 1
            #
            #         if min_angle_dist < self._spatial_T:
            #             self._num_correct_pred += 1
            #             ### Compute Precision and recall
            #             self._TP += 1
            #
            #             if num_of_gt_ref == 1:
            #                 self._num_correct_pred_Srcs[0] += 1
            #             elif num_of_gt_ref == 2:
            #                 self._num_correct_pred_Srcs[1] += 1
            #
            #         # if min_angle_dist < self._spatial_T:
            #         #     self._num_correct_pred += 1
            #         #     ### Compute Precision and recall
            #         #     if tmp_pred_doa_value > self._pred_T:
            #         #         self._TP += 1
            #         # if tmp_pred_doa_value > self._pred_T:
            #         #     self._num_all_pred += 1
            #     else:
            #         print("There is some issues!!! Check it ")


    # def update_seld_scores_wo_training(self, num_fr, gt_doas, pred):
    #
    #     for idx_fr in range(num_fr):
    #
    #
    #         num_of_gt_ref = int(gt_doas[idx_fr, 0])
    #         num_of_gt_pred_th = int(pred[idx_fr, 0])
    #
    #         #########
    #         ### Unknown the number of sources (Precision & Recall)
    #         ### Denoms of Precision and recall
    #         self._num_all_gts += num_of_gt_ref
    #         self._num_all_pred += num_of_gt_pred_th
    #
    #         ### Compute Distance
    #         # for idx_gt_pred in range(num_of_gt_pred):
    #         for idx_gt_pred in range(min(num_of_gt_pred_th, num_of_gt_ref)):
    #             # tmp_pred_doa_value = peak_list_th[idx_gt_pred][1]
    #             # tmp_pred_doa_idx = int(peak_list_th[idx_gt_pred][0])
    #             tmp_pred_doa = pred[idx_fr, 1 + idx_gt_pred * 3: 1 + (idx_gt_pred + 1) * 3]
    #
    #             min_angle_dist = 100
    #             for idx_gt_ref in range(num_of_gt_ref):
    #                 # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
    #                 tmp_gt_doa = gt_doas[idx_fr, 1 + idx_gt_ref * 3: 1 + (idx_gt_ref + 1) * 3]
    #
    #                 angle_dist = abs(azimuth_distance(tmp_pred_doa, tmp_gt_doa))
    #
    #                 if min_angle_dist > angle_dist:
    #                     min_angle_dist = angle_dist
    #
    #             if min_angle_dist < 99:  # Check exception condition
    #                 if min_angle_dist < self._spatial_T:
    #                     ### Compute Precision and recall
    #                     self._TP += 1
    #
    #             else:
    #                 print("There is some issues!!! Check it ")
    #
    #         #########
    #         ### Known the number of sources (Precision & Recall)
    #         ### Denoms of Precision and recall
    #         # for idx_gt_pred in range(num_of_gt_ref):  # Backup
    #
    #         for idx_gt_pred in range(min(num_of_gt_pred_th, num_of_gt_ref)):
    #             # tmp_pred_doa_value = peak_list[idx_gt_pred][1]
    #             # tmp_pred_doa_idx = int(peak_list[idx_gt_pred][0])
    #             tmp_pred_doa = pred[idx_fr, 1 + idx_gt_pred * 3: 1 + (idx_gt_pred + 1) * 3]
    #
    #             min_angle_dist = 100
    #             for idx_gt_ref in range(num_of_gt_ref):
    #                 # tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]
    #                 tmp_gt_doa = gt_doas[idx_fr, 1 + idx_gt_ref * 3: 1 + (idx_gt_ref + 1) * 3]
    #
    #                 angle_dist = abs(azimuth_distance(tmp_pred_doa, tmp_gt_doa))
    #
    #                 if min_angle_dist > angle_dist:
    #                     min_angle_dist = angle_dist
    #
    #             if min_angle_dist < 99:  # Check exception condition
    #                 self._total_DE += min_angle_dist
    #                 self._num_all_peaks += 1
    #                 if num_of_gt_ref == 1:
    #                     self._total_DE_Srcs[0] += min_angle_dist
    #                     self._num_all_peaks_Srcs[0] += 1
    #                     # self._DE_TP_Srcs[0] += 1
    #                 elif num_of_gt_ref == 2:
    #                     self._total_DE_Srcs[1] += min_angle_dist
    #                     self._num_all_peaks_Srcs[1] += 1
    #                     # self._DE_TP_Srcs[1] += 1
    #
    #                 if min_angle_dist < self._spatial_T:
    #                     self._num_correct_pred += 1
    #
    #                     if num_of_gt_ref == 1:
    #                         self._num_correct_pred_Srcs[0] += 1
    #                     elif num_of_gt_ref == 2:
    #                         self._num_correct_pred_Srcs[1] += 1
    #
    #             else:
    #                 print("There is some issues!!! Check it ")
