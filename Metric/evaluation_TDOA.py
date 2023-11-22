import sys
import os
import numpy as np

from scipy.signal import find_peaks
from scipy.spatial import distance
import itertools

import matplotlib.pyplot as plt

eps = np.finfo(np.float64).eps

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def decoding_pred(pred_fr, half_delay_len):
    delay_len = 2 * half_delay_len + 1
    peak_list = []
    neighbor_delays = [-3, -2, -1, 1, 2, 3]
    threshold_pred = 0.2

    for idx_delay in range(delay_len):
        if pred_fr[idx_delay] > threshold_pred:
            is_peak = True
            for delay in neighbor_delays:
                idx_neigh_delay = delay + idx_delay
                if idx_neigh_delay >= 0 and idx_neigh_delay < delay_len:
                    if pred_fr[idx_delay] < pred_fr[idx_neigh_delay]:
                        is_peak = False
            if is_peak:
                peak_list.append((idx_delay, pred_fr[idx_delay]))
    peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
    return peak_list


######################################################################################################
class evaluation_tdoa:
    def __init__(self, half_delay_len=25, pred_threshold=0.2, w_pred_threshold=False):
        self._half_delay_len = half_delay_len
        self._delay_len = half_delay_len * 2 + 1

        self._sum_diff = 0
        self._sum_peak_err = 0
        self._Nref = 0

        if w_pred_threshold:
            self._pred_T = pred_threshold
        else:
            self._pred_T = 0

        self._DE_TP = 0
        self._DE_FP = 0
        self._DE_FN = 0

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # TDOA error
        total_TDOA_error = self._sum_diff / self._Nref
        # Peak coeff. error
        total_Peak_error = self._sum_peak_err / self._Nref
        # TDOA recall
        recall = self._DE_TP / (self._DE_TP + self._DE_FN)
        # TDOA precision
        precision = self._DE_TP / (self._DE_TP + self._DE_FP)

        return total_TDOA_error, total_Peak_error, recall, precision

    def early_stopping_metric(self, tdoa_error, peak_error):

        total_error = np.mean([
            tdoa_error, peak_error
        ])

        return total_error

    def update_scores(self, preds, gts):
        num_bat, num_fr, num_delay = preds.shape

        for idx_bat in range(num_bat):
            for idx_fr in range(num_fr):
                loc_FN, loc_FP = 0, 0

                # Compute peaks
                peak_indice = find_peaks(preds[idx_bat, idx_fr, :])

                peak_list = np.array([preds[idx_bat, idx_fr, peak_indice[0]], peak_indice[0]]).transpose().tolist()
                peak_list = sorted(peak_list, key=lambda peak: peak[0], reverse=True)
                peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[0]]

                # plt.figure()
                # plt.plot(range(self._delay_len), preds[idx_bat, idx_fr, :])
                # plt.show()

                # Num of GT
                num_of_gt_ref = int(gts[idx_bat, idx_fr, 0])
                # Num of Peaks
                num_of_gt_peaks = len(peak_list)

                self._Nref += num_of_gt_ref

                ### GT is one at a single frame
                if num_of_gt_ref == 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_index = gts[idx_bat, idx_fr, 1]
                        tdoa_preds_index = tdoa_preds[0, 1]
                        tdoa_preds_coeff = tdoa_preds[0, 0]

                        # tmp_diff = abs(tdoa_gts_index - tdoa_preds_index)
                        # tmp_peak_err = 1 - tdoa_preds_coeff

                        self._sum_diff += abs(tdoa_gts_index - tdoa_preds_index)
                        self._sum_peak_err += (1 - tdoa_preds_coeff)
                        self._DE_TP += 1
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # There is any peak
                        self._sum_diff += self._half_delay_len
                        self._sum_peak_err += 1
                        self._DE_FN += 1
                ### GTs are larger than one at a single frame
                elif num_of_gt_ref > 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_indice = gts[idx_bat, idx_fr, 1:num_of_gt_ref + 1]

                        tdoa_preds_indice = np.expand_dims(tdoa_preds[:, 1], axis=-1)
                        tdoa_preds_coeff = tdoa_preds[:, 0]
                        tdoa_gts_indice = np.expand_dims(tdoa_gts_indice, axis=-1)

                        # Compute the distance of TDOAs
                        diff_tdoa_mat = distance.cdist(tdoa_gts_indice, tdoa_preds_indice, metric='minkowski', p=1)

                        diff_per_gts = np.min(diff_tdoa_mat, axis=1)
                        selected_peak_indice = np.argmin(diff_tdoa_mat, axis=1)

                        self._sum_diff += np.sum(diff_per_gts)
                        self._sum_peak_err += np.sum(1 - tdoa_preds_coeff[selected_peak_indice])
                        self._DE_TP += num_of_gt_ref
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # If there is any peak
                        self._sum_diff += self._half_delay_len * num_of_gt_ref
                        self._sum_peak_err += num_of_gt_ref
                        self._DE_FN += num_of_gt_ref
                # There is no GT
                else:
                    self._DE_FP += num_of_gt_peaks


class evaluation_tdoa_multi_labels:
    def __init__(self, half_delay_len=25, pred_threshold=0.2, w_pred_threshold=False):
        self._half_delay_len = half_delay_len
        self._delay_len = half_delay_len * 2 + 1

        self._sum_diff = 0
        self._sum_peak_err = 0
        self._Nref = 0

        if w_pred_threshold:
            self._pred_T = pred_threshold
        else:
            self._pred_T = 0

        self._DE_TP = 0
        self._DE_FP = 0
        self._DE_FN = 0

        self._idx_speech = 5

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # TDOA error
        total_TDOA_error = self._sum_diff / self._Nref
        # Peak coeff. error
        total_Peak_error = self._sum_peak_err / self._Nref
        # TDOA recall
        recall = self._DE_TP / (self._DE_TP + self._DE_FN)
        # TDOA precision
        precision = self._DE_TP / (self._DE_TP + self._DE_FP)

        return total_TDOA_error, total_Peak_error, recall, precision

    def early_stopping_metric(self, tdoa_error, peak_error):

        total_error = np.mean([
            tdoa_error, peak_error
        ])

        return total_error

    def update_scores(self, preds, gts):
        num_bat, num_labels, num_fr, num_delay = preds.shape

        for idx_label in range(num_labels):
            # Extract predictions and gts of each label
            # tmp_preds = preds[:, self._idx_speech]
            # tmp_gts = gts[:, self._idx_speech]
            tmp_preds = preds[:, idx_label]
            tmp_gts = gts[:, idx_label]

            for idx_bat in range(num_bat):
                for idx_fr in range(num_fr):

                    # Num of GT
                    num_of_gt_ref = int(tmp_gts[idx_bat, idx_fr, 0])
                    if num_of_gt_ref > 0:
                        # Compute peaks
                        peak_indice = find_peaks(tmp_preds[idx_bat, idx_fr, :])

                        peak_list = np.array([tmp_preds[idx_bat, idx_fr, peak_indice[0]], peak_indice[0]]).transpose().tolist()
                        peak_list = sorted(peak_list, key=lambda peak: peak[0], reverse=True)
                        peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[0]]

                        # plt.figure()
                        # plt.plot(range(self._delay_len), preds[idx_bat, idx_fr, :])
                        # plt.show()

                        # Num of Peaks
                        num_of_gt_peaks = len(peak_list)

                        self._Nref += num_of_gt_ref

                        ### GT is one at a single frame
                        if num_of_gt_ref == 1:
                            if num_of_gt_peaks > 0: # If there is peaks
                                tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                                tdoa_gts_index = tmp_gts[idx_bat, idx_fr, 1]
                                tdoa_preds_index = tdoa_preds[0, 1]
                                tdoa_preds_coeff = tdoa_preds[0, 0]

                                # tmp_diff = abs(tdoa_gts_index - tdoa_preds_index)
                                # tmp_peak_err = 1 - tdoa_preds_coeff

                                self._sum_diff += abs(tdoa_gts_index - tdoa_preds_index)
                                self._sum_peak_err += (1 - tdoa_preds_coeff)
                                self._DE_TP += 1
                                self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                            else:   # There is any peak
                                self._sum_diff += self._half_delay_len
                                self._sum_peak_err += 1
                                self._DE_FN += 1
                        ### GTs are larger than one at a single frame
                        elif num_of_gt_ref > 1:
                            if num_of_gt_peaks > 0: # If there is peaks
                                tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                                tdoa_gts_indice = tmp_gts[idx_bat, idx_fr, 1:num_of_gt_ref + 1]

                                tdoa_preds_indice = np.expand_dims(tdoa_preds[:, 1], axis=-1)
                                tdoa_preds_coeff = tdoa_preds[:, 0]
                                tdoa_gts_indice = np.expand_dims(tdoa_gts_indice, axis=-1)

                                # Compute the distance of TDOAs
                                diff_tdoa_mat = distance.cdist(tdoa_gts_indice, tdoa_preds_indice, metric='minkowski', p=1)

                                diff_per_gts = np.min(diff_tdoa_mat, axis=1)
                                selected_peak_indice = np.argmin(diff_tdoa_mat, axis=1)

                                self._sum_diff += np.sum(diff_per_gts)
                                self._sum_peak_err += np.sum(1 - tdoa_preds_coeff[selected_peak_indice])
                                self._DE_TP += num_of_gt_ref
                                self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                            else:   # If there is any peak
                                self._sum_diff += self._half_delay_len * num_of_gt_ref
                                self._sum_peak_err += num_of_gt_ref
                                self._DE_FN += num_of_gt_ref
                        # There is no GT
                        else:
                            self._DE_FP += num_of_gt_peaks


class evaluation_tdoa_speech_label:
    def __init__(self, half_delay_len=25, pred_threshold=0.2, w_pred_threshold=False):
        self._half_delay_len = half_delay_len
        self._delay_len = half_delay_len * 2 + 1

        self._sum_diff = 0
        self._sum_peak_err = 0
        self._Nref = 0

        if w_pred_threshold:
            self._pred_T = pred_threshold
        else:
            self._pred_T = 0

        self._DE_TP = 0
        self._DE_FP = 0
        self._DE_FN = 0

        self._idx_speech = 5

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # TDOA error
        # total_TDOA_error = self._sum_diff / self._Nref
        total_TDOA_error = 100. if self._Nref <= 0. else self._sum_diff / self._Nref
        # Peak coeff. error
        total_Peak_error = 1. if self._Nref <= 0. else self._sum_peak_err / self._Nref
        # TDOA recall
        recall = 0. if self._DE_TP + self._DE_FN <= 0. else self._DE_TP / (self._DE_TP + self._DE_FN)
        # TDOA precision
        precision = 0. if self._DE_TP + self._DE_FP <= 0. else self._DE_TP / (self._DE_TP + self._DE_FP)

        return total_TDOA_error, total_Peak_error, recall, precision

    def early_stopping_metric(self, tdoa_error, peak_error):

        total_error = np.mean([
            tdoa_error, peak_error
        ])

        return total_error

    def update_scores(self, preds, gts):
        num_bat, num_labels, num_fr, num_delay = preds.shape

        # Extract predictions and gts of each label
        tmp_preds = preds[:, self._idx_speech]
        tmp_gts = gts[:, self._idx_speech]

        for idx_bat in range(num_bat):
            for idx_fr in range(num_fr):
                loc_FN, loc_FP = 0, 0

                # Compute peaks
                peak_indice = find_peaks(tmp_preds[idx_bat, idx_fr, :])

                peak_list = np.array([tmp_preds[idx_bat, idx_fr, peak_indice[0]], peak_indice[0]]).transpose().tolist()
                peak_list = sorted(peak_list, key=lambda peak: peak[0], reverse=True)
                peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[0]]

                # plt.figure()
                # plt.plot(range(self._delay_len), preds[idx_bat, idx_fr, :])
                # plt.show()

                # Num of GT
                num_of_gt_ref = int(tmp_gts[idx_bat, idx_fr, 0])
                # Num of Peaks
                num_of_gt_peaks = len(peak_list)

                self._Nref += num_of_gt_ref

                ### GT is one at a single frame
                if num_of_gt_ref == 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_index = tmp_gts[idx_bat, idx_fr, 1]
                        tdoa_preds_index = tdoa_preds[0, 1]
                        tdoa_preds_coeff = tdoa_preds[0, 0]

                        # tmp_diff = abs(tdoa_gts_index - tdoa_preds_index)
                        # tmp_peak_err = 1 - tdoa_preds_coeff

                        self._sum_diff += abs(tdoa_gts_index - tdoa_preds_index)
                        self._sum_peak_err += (1 - tdoa_preds_coeff)
                        self._DE_TP += 1
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # There is any peak
                        self._sum_diff += self._half_delay_len
                        self._sum_peak_err += 1
                        self._DE_FN += 1
                ### GTs are larger than one at a single frame
                elif num_of_gt_ref > 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_indice = tmp_gts[idx_bat, idx_fr, 1:num_of_gt_ref + 1]

                        tdoa_preds_indice = np.expand_dims(tdoa_preds[:, 1], axis=-1)
                        tdoa_preds_coeff = tdoa_preds[:, 0]
                        tdoa_gts_indice = np.expand_dims(tdoa_gts_indice, axis=-1)

                        # Compute the distance of TDOAs
                        diff_tdoa_mat = distance.cdist(tdoa_gts_indice, tdoa_preds_indice, metric='minkowski', p=1)

                        diff_per_gts = np.min(diff_tdoa_mat, axis=1)
                        selected_peak_indice = np.argmin(diff_tdoa_mat, axis=1)

                        self._sum_diff += np.sum(diff_per_gts)
                        self._sum_peak_err += np.sum(1 - tdoa_preds_coeff[selected_peak_indice])
                        self._DE_TP += num_of_gt_ref
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # If there is any peak
                        self._sum_diff += self._half_delay_len * num_of_gt_ref
                        self._sum_peak_err += num_of_gt_ref
                        self._DE_FN += num_of_gt_ref
                # There is no GT
                else:
                    self._DE_FP += num_of_gt_peaks

################################################################################################################
### Backup, 20220104

class evaluation_tdoa_multi_labels_backup:
    def __init__(self, half_delay_len=25, pred_threshold=0.2, w_pred_threshold=False):
        self._half_delay_len = half_delay_len
        self._delay_len = half_delay_len * 2 + 1

        self._sum_diff = 0
        self._sum_peak_err = 0
        self._Nref = 0

        if w_pred_threshold:
            self._pred_T = pred_threshold
        else:
            self._pred_T = 0

        self._DE_TP = 0
        self._DE_FP = 0
        self._DE_FN = 0

        self._idx_speech = 5

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # TDOA error
        total_TDOA_error = self._sum_diff / self._Nref
        # Peak coeff. error
        total_Peak_error = self._sum_peak_err / self._Nref
        # TDOA recall
        recall = self._DE_TP / (self._DE_TP + self._DE_FN)
        # TDOA precision
        precision = self._DE_TP / (self._DE_TP + self._DE_FP)

        return total_TDOA_error, total_Peak_error, recall, precision

    def early_stopping_metric(self, tdoa_error, peak_error):

        total_error = np.mean([
            tdoa_error, peak_error
        ])

        return total_error

    def update_scores(self, preds, gts):
        num_bat, _, num_fr, num_delay = preds.shape

        # Extract speech predictions and gts
        preds = preds[:, self._idx_speech]
        gts = gts[:, self._idx_speech]

        for idx_bat in range(num_bat):
            for idx_fr in range(num_fr):
                loc_FN, loc_FP = 0, 0

                # Compute peaks
                peak_indice = find_peaks(preds[idx_bat, idx_fr, :])

                peak_list = np.array([preds[idx_bat, idx_fr, peak_indice[0]], peak_indice[0]]).transpose().tolist()
                peak_list = sorted(peak_list, key=lambda peak: peak[0], reverse=True)
                peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[0]]

                # plt.figure()
                # plt.plot(range(self._delay_len), preds[idx_bat, idx_fr, :])
                # plt.show()

                # Num of GT
                num_of_gt_ref = int(gts[idx_bat, idx_fr, 0])
                # Num of Peaks
                num_of_gt_peaks = len(peak_list)

                self._Nref += num_of_gt_ref

                ### GT is one at a single frame
                if num_of_gt_ref == 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_index = gts[idx_bat, idx_fr, 1]
                        tdoa_preds_index = tdoa_preds[0, 1]
                        tdoa_preds_coeff = tdoa_preds[0, 0]

                        # tmp_diff = abs(tdoa_gts_index - tdoa_preds_index)
                        # tmp_peak_err = 1 - tdoa_preds_coeff

                        self._sum_diff += abs(tdoa_gts_index - tdoa_preds_index)
                        self._sum_peak_err += (1 - tdoa_preds_coeff)
                        self._DE_TP += 1
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # There is any peak
                        self._sum_diff += self._half_delay_len
                        self._sum_peak_err += 1
                        self._DE_FN += 1
                ### GTs are larger than one at a single frame
                elif num_of_gt_ref > 1:
                    if num_of_gt_peaks > 0: # If there is peaks
                        tdoa_preds = np.array(peak_list[:num_of_gt_ref])
                        tdoa_gts_indice = gts[idx_bat, idx_fr, 1:num_of_gt_ref + 1]

                        tdoa_preds_indice = np.expand_dims(tdoa_preds[:, 1], axis=-1)
                        tdoa_preds_coeff = tdoa_preds[:, 0]
                        tdoa_gts_indice = np.expand_dims(tdoa_gts_indice, axis=-1)

                        # Compute the distance of TDOAs
                        diff_tdoa_mat = distance.cdist(tdoa_gts_indice, tdoa_preds_indice, metric='minkowski', p=1)

                        diff_per_gts = np.min(diff_tdoa_mat, axis=1)
                        selected_peak_indice = np.argmin(diff_tdoa_mat, axis=1)

                        self._sum_diff += np.sum(diff_per_gts)
                        self._sum_peak_err += np.sum(1 - tdoa_preds_coeff[selected_peak_indice])
                        self._DE_TP += num_of_gt_ref
                        self._DE_FP += (num_of_gt_peaks - num_of_gt_ref)
                    else:   # If there is any peak
                        self._sum_diff += self._half_delay_len * num_of_gt_ref
                        self._sum_peak_err += num_of_gt_ref
                        self._DE_FN += num_of_gt_ref
                # There is no GT
                else:
                    self._DE_FP += num_of_gt_peaks

################################################################################################################



class evaluation_SELD_multi_labels:
    def __init__(self, half_delay_len=25, delay_threshold=2, pred_threshold=0.2):
        self._half_delay_len = half_delay_len
        self._delay_len = half_delay_len * 2 + 1

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

        # self._spatial_T = doa_threshold * np.pi / 180
        self._spatial_T = delay_threshold
        self._pred_T = pred_threshold

        # variables for DoAs
        self._total_DE = 0
        self._total_DE_Srcs = [0, 0]

        self._DE_TP = 0
        self._DE_TP_Srcs = [0, 0]
        self._DE_FP = 0
        self._DE_FN = 0

        self._idx_speech = 5

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
            self._DE_TP + eps) if self._DE_TP else self._half_delay_len  # When the total number of prediction is zero
        LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)
        # print('S {}, D {}, I {}, Nref {}, TP {}, FP {}, FN {}, DE_TP {}, DE_FN {}, totalDE {}'.format(self._S, self._D, self._I, self._Nref, self._TP, self._FP, self._FN, self._DE_TP, self._DE_FN, self._total_DE))
        return ER, F, LE, LR

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
            doa_error[0] / float(self._half_delay_len),
            1 - doa_error[1]]
        )
        return seld_metric

    def decoding_pred(self, pred_fr):

        peak_list = []
        neighbor_delays = [-3, -2, -1, 1, 2, 3]
        # threshold_pred = 0.2

        for idx_delay in range(self._delay_len):
            if pred_fr[idx_delay] > self._pred_T:
                is_peak = True
                for delay in neighbor_delays:
                    idx_neigh_delay = delay + idx_delay
                    if idx_neigh_delay >= 0 and idx_neigh_delay < self._delay_len:
                        if pred_fr[idx_delay] < pred_fr[idx_neigh_delay]:
                            is_peak = False
                if is_peak:
                    peak_list.append((idx_delay, pred_fr[idx_delay]))
        peak_list = sorted(peak_list, key=lambda peak: peak[1], reverse=True)
        return peak_list

    # def evaluation_MAE(self, nFrame, num_of_gt, gt_doas, pred, threshold=0.2):
    def update_seld_scores(self, preds, gts):
        # num_bat, num_fr, num_delay = preds.shape
        num_bat, _, num_fr, num_delay = preds.shape

        # Extract speech predictions and gts
        preds = preds[:, self._idx_speech]
        gts = gts[:, self._idx_speech]

        for idx_bat in range(num_bat):
            for idx_fr in range(num_fr):
                loc_FN, loc_FP = 0, 0

                peak_list = self.decoding_pred(preds[idx_bat, idx_fr, :])
                # Compute peaks
                # peak_indice = find_peaks(preds[idx_bat, idx_fr, :])
                #
                # peak_list = np.array([preds[idx_bat, idx_fr, peak_indice[0]], peak_indice[0]]).transpose().tolist()
                # peak_list = sorted(peak_list, key=lambda peak: peak[0], reverse=True)
                # peak_list = [tmp_peak for tmp_peak in peak_list if self._pred_T < tmp_peak[0]]

                num_of_gt_ref = int(gts[idx_bat, idx_fr, 0])
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

                ### Compute Distance
                # for idx_gt_pred in range(num_of_gt_pred):
                for idx_gt_pred in range(min(num_of_gt_pred, num_of_gt_ref)):
                    tmp_pred_delay = peak_list[idx_gt_pred][0]

                    min_delay_diff = 100
                    for idx_gt_ref in range(num_of_gt_ref):
                        tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]

                        delay_diff = abs(tmp_gt_delay - tmp_pred_delay)

                        if min_delay_diff > delay_diff:
                            min_delay_diff = delay_diff
                            # self._total_DE += delay_diff
                            # self._total_DE_Srcs[0] += delay_diff
                    if min_delay_diff < 99:  # Check exception condition
                        self._total_DE += min_delay_diff
                        if num_of_gt_ref == 1:
                            self._total_DE_Srcs[0] += min_delay_diff
                            self._DE_TP_Srcs[0] += 1
                        elif num_of_gt_ref == 2:
                            self._total_DE_Srcs[1] += min_delay_diff
                            self._DE_TP_Srcs[1] += 1
                    else:
                        print("There is some issues!!! Check it ")

                # TP_FN_array[i] == 1: TP, TP_FN_array[i] == 0: FN
                TP_FN_array = np.zeros(num_of_gt_ref)
                # FP_array[i] == 1: FP, TP_FN_array[i] == 0: TP or TPs
                FP_array = np.zeros(num_of_gt_pred)
                for idx_gt_ref in range(num_of_gt_ref):
                    tmp_gt_delay = gts[idx_bat, idx_fr, idx_gt_ref + 1]

                    for idx_gt_pred in range(num_of_gt_pred):
                        tmp_pred_delay = peak_list[idx_gt_pred][0]

                        delay_diff = abs(tmp_gt_delay - tmp_pred_delay)

                        if delay_diff < self._spatial_T:
                            # TP
                            TP_FN_array[idx_gt_ref] = 1
                        else:
                            # FP
                            FP_array[idx_gt_pred] = 1


                ### Compute SED score
                for idx_gt_ref in range(num_of_gt_ref):
                    if TP_FN_array[idx_gt_ref] == 1:    # TP
                        self._TP += 1
                    else:       # FN
                        self._FN += 1
                        loc_FN += 1

                for idx_gt_pred in range(num_of_gt_pred):
                    if FP_array[idx_gt_pred] == 1:  # FP
                        self._FP += 1
                        loc_FP += 1

                self._S += np.minimum(loc_FP, loc_FN)
                self._D += np.maximum(0, loc_FN - loc_FP)
                self._I += np.maximum(0, loc_FP - loc_FN)

