import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import utils.utils as utils
from apkit.apkit.doa import load_pts_horizontal

# class Angle:
#     def __init__(self, degree, depth, fs=24000):
#         self.fs = fs
#         self.depth = depth
#         self.steer_angle = degree   # Degree: 0 ~ 360
#         self.steer_range = []  # Consists of two end points
#         self.child_indice = []
#         self.TDOA_range = []
#
#         angle_window = 90
#         for idx_depth in range(depth):
#             angle_window = angle_window / 2.0
#         half_angle_window = angle_window / 2.0
#         min_steer_range = self.steer_angle - half_angle_window if self.steer_angle - half_angle_window > 0 else 0
#         max_steer_range = self.steer_angle + half_angle_window if self.steer_angle + half_angle_window < 360 else 360
#         self.steer_range.append(min_steer_range)
#         self.steer_range.append(max_steer_range)
#
#     def set_child(self, input_idx):
#         if input_idx not in self.child_indice:
#             self.child_indice.append(input_idx)
#
#     def get_coord(self):
#         return self.deg2coord(self.steer_angle)
#
#     def deg2rad(self, degree):
#         return degree * np.pi / 180.0
#
#     def deg2coord(self, degree):
#         rad = self.deg2rad(degree)
#         return np.array([np.cos(rad), np.sin(rad), 0])
#
#     def compute_TDOA_range(self, mic_pair_pos, half_delay_len):
#         # Extract neighbor vertices
#         range_angle_list = [self.deg2coord(range_angle) for range_angle in self.steer_range]
#
#         # Compute ranges of TDOAs of neighbors
#         range_angle_diff_list = utils.utils.compute_diff_TDOA(range_angle_list, np.array([0, 0, 0]),
#                                                                         mic_pair_pos, half_delay_len,
#                                                               sampling_rate=self.fs)
#
#         self.TDOA_range = np.stack((np.min(range_angle_diff_list, axis=0), np.max(range_angle_diff_list, axis=0)),
#                                    axis=-1).tolist()
#         # self.TDOA_range = np.stack((np.min(range_angle_diff_list, axis=0)-1, np.max(range_angle_diff_list, axis=0)+1),
#         #                            axis=-1).tolist()

class CircleGrid:
    # def __init__(self, subdiv, fs=24000, DOA_likelihood_threshold=0.6):
    def __init__(self, fs=24000, z_angle=0):
        self.fs = fs
        # self.DOA_likelihood_threshold = DOA_likelihood_threshold

        #######################################
        # Version 1
        # init_steer_angles = [45, 135, 225, 315] # Depth 0
        #
        # init_angles = []
        # for tmp_angle in init_steer_angles:
        #     init_angles.append(Angle(tmp_angle, 0, fs=self.fs))
        #
        # # Initialize the empty lists of angles for each depth
        # self.steer_angle_list = [[] for i in range(subdiv + 1)]
        # self.steer_angle_list[0] = init_angles
        #
        # # Perform dividing each angle ranges into two angle ranges
        # self.initialize_recursive_circle(subdiv)

        #######################################
        # Version 2
        self.pts_horizontal = load_pts_horizontal()

        self.pts_horizontal[:, 2] = np.sqrt(self.pts_horizontal[:, 0]*self.pts_horizontal[:, 0] + self.pts_horizontal[:, 1]*self.pts_horizontal[:, 1]) * np.tan(z_angle*np.pi/180)
        self.pts_horizontal[:, 0] = self.pts_horizontal[:, 0] / np.linalg.norm(self.pts_horizontal, axis=-1)
        self.pts_horizontal[:, 1] = self.pts_horizontal[:, 1] / np.linalg.norm(self.pts_horizontal, axis=-1)
        self.pts_horizontal[:, 2] = self.pts_horizontal[:, 2] / np.linalg.norm(self.pts_horizontal, axis=-1)


        self.sigma_gauss_dist = 1.5
        self.TDOA_filter = None
        # print("Debugging")

    # def estimate_DOA(self, tdoa_pred):
    # def estimate_DOA(self, tdoa_pred, gt_doas): # for testing
    #     # E.g., [128, 6, 10, 21] --> The shape of GT DOAs: [128, 10, 9 (DOAs) + 1]
    #     batch_size, num_mic_pairs, num_frames, delay_len = tdoa_pred.shape
    #
    #     # The output: DOA predictions (Shape: 128, 10, 9 + 1)
    #     doas_pred = np.zeros((batch_size, num_frames, 10))
    #
    #     # For all batches
    #     for batch_idx in range(batch_size):
    #         # For all frames
    #         for frame_idx in range(num_frames):
    #             # Empty list of estimated DOAs
    #             estimated_DOAs = []
    #             computed_DOAs_list = []
    #             # estimated_DOAs_coord = []
    #
    #             tmp_tdoa_pred = tdoa_pred[batch_idx, :, frame_idx, :]
    #
    #             # For the deepest depth
    #             steer_angles = self.steer_angle_list[-1]
    #
    #             # For all angles
    #             for idx_angle, tmp_angle in enumerate(steer_angles):
    #                 # Extract the TDOA range of the steering angle
    #                 angle_TDOA_range = tmp_angle.TDOA_range
    #
    #                 # Compute the DOA likelihood
    #                 DOA_likelihood_list = []
    #                 for idx_pair, (range_min, range_max) in enumerate(angle_TDOA_range):
    #                     tmp_TDOA_likelihoold = np.max(tmp_tdoa_pred[idx_pair, range_min: range_max + 1], axis=-1)
    #                     DOA_likelihood_list.append(tmp_TDOA_likelihoold)
    #                 # Compute the mean of the DOA likelihood of all mic's pairs
    #                 avg_DOA_likelihood = np.mean(np.array(DOA_likelihood_list), axis=0)
    #
    #                 computed_DOAs_list.append((avg_DOA_likelihood, idx_angle, tmp_angle))
    #                 # If the DOA likelihood is larger than the DOA likelihood,
    #                 if avg_DOA_likelihood > self.DOA_likelihood_threshold:
    #                     estimated_DOAs.append((avg_DOA_likelihood, idx_angle, tmp_angle))
    #                     # estimated_DOAs_coord.append(tmp_angle.get_coord())
    #
    #             # Consider a single source, 20220608 IK
    #             max_DOAs = max(computed_DOAs_list, key= lambda DOAs: DOAs[0])
    #             if max_DOAs[0] > self.DOA_likelihood_threshold:
    #                 tmp_angle = max_DOAs[2]
    #                 estimated_DOA = tmp_angle.get_coord()
    #                 idx_DOA = 0
    #                 doas_pred[batch_idx, frame_idx, 0] = 1
    #                 doas_pred[batch_idx, frame_idx, 1 + idx_DOA * 3:1 + idx_DOA * 3 + 3] = estimated_DOA
    #
    #             # Consider multiple sources, 20220608 IK
    #             # # If there exist esimated DOAs,
    #             # if len(estimated_DOAs) > 0:
    #             #
    #             #     # Check the continuous between the zero and last angle
    #             #     if estimated_DOAs[0][1] == 0:
    #             #         TDOA_likelihood, idx_angle, tmp_angle = estimated_DOAs.pop(0)
    #             #         estimated_DOAs.append((TDOA_likelihood, idx_angle, tmp_angle))
    #             #         while idx_angle + 1 == estimated_DOAs[0][1]:
    #             #             TDOA_likelihood, idx_angle, tmp_angle = estimated_DOAs.pop(0)
    #             #             estimated_DOAs.append((TDOA_likelihood, idx_angle, tmp_angle))
    #             #
    #             #     # Grouping the neighbors of estimated DOAs
    #             #     # Grouped DOAs become the final DOAs
    #             #     starting_flg = True
    #             #     prev_idx = 0
    #             #     estimated_DOAs_group_list = []  # It contains the grouped DOAs
    #             #     estimated_DOAs_group = []
    #             #     DOA_likelihood_list = []
    #             #     for idx_DOA, (DOA_likelihood, idx_angle, tmp_angle) in enumerate(estimated_DOAs):
    #             #         if starting_flg:
    #             #             prev_idx = idx_angle
    #             #             estimated_DOAs_group.clear()
    #             #             estimated_DOAs_group.append(tmp_angle.get_coord())
    #             #             DOA_likelihood_list.append(DOA_likelihood)
    #             #             starting_flg = False
    #             #         else:
    #             #             if prev_idx + 1 == idx_angle:
    #             #                 prev_idx = idx_angle
    #             #                 estimated_DOAs_group.append(tmp_angle.get_coord())
    #             #                 DOA_likelihood_list.append(DOA_likelihood)
    #             #             elif prev_idx == len(steer_angles) - 1 and idx_angle == 0:
    #             #                 prev_idx = idx_angle
    #             #                 estimated_DOAs_group.append(tmp_angle.get_coord())
    #             #                 DOA_likelihood_list.append(DOA_likelihood)
    #             #             else:
    #             #                 starting_flg = True
    #             #                 estimated_DOAs_group_list.append((
    #             #                     np.mean(np.array(estimated_DOAs_group), axis=0),
    #             #                     np.mean(np.array(DOA_likelihood_list))
    #             #                 ))
    #             #
    #             #         if idx_DOA == len(estimated_DOAs) - 1 and len(estimated_DOAs_group) > 0:
    #             #             # estimated_DOAs_group_list.append(np.mean(np.array(estimated_DOAs_group), axis=0))
    #             #             estimated_DOAs_group_list.append((
    #             #                 np.mean(np.array(estimated_DOAs_group), axis=0),
    #             #                 np.mean(np.array(DOA_likelihood_list))
    #             #             ))
    #             #
    #             #     # Set the final output ...
    #             #     if len(estimated_DOAs_group_list) > 0:
    #             #         # if len(estimated_DOAs_group_list) > 1:
    #             #         #     print("Check point")
    #             #
    #             #         estimated_DOAs_group_list.sort(key=lambda tmp_DOA_info: tmp_DOA_info[1], reverse=True)
    #             #
    #             #         doas_pred[batch_idx, frame_idx, 0] = len(estimated_DOAs_group_list)
    #             #         for idx_DOA, (estimated_DOA, avg_DOA_likelihood) in enumerate(estimated_DOAs_group_list):
    #             #             if idx_DOA < 3:
    #             #                 doas_pred[batch_idx, frame_idx, 1+idx_DOA*3:1+idx_DOA*3+3] = estimated_DOA
    #     return doas_pred

    # 20220527, backup
    # def estimate_DOA(self, tdoa_pred):
    #     # E.g., [128, 6, 10, 21] --> The shape of GT DOAs: [128, 9 (DOAs), 10]
    #     batch_size, num_mic_pairs, num_frames, delay_len = tdoa_pred.shape
    #
    #     for batch_idx in range(batch_size):
    #         for frame_idx in range(num_frames):
    #             estimated_DOAs = []
    #
    #             tmp_tdoa_pred = tdoa_pred[batch_idx, :, frame_idx, :]
    #             # For depth 0
    #             steer_angles = self.steer_angle_list[0]
    #
    #             # avg_TDOA_likelihood_list = []
    #             for tmp_angle in steer_angles:
    #                 angle_TDOA_range = tmp_angle.TDOA_range
    #
    #                 TDOA_likelihood_list = []
    #                 for idx_pair, (range_min, range_max) in enumerate(angle_TDOA_range):
    #                     tmp_TDOA_likelihoold = np.max(tmp_tdoa_pred[idx_pair, range_min: range_max + 1], axis=-1)
    #                     TDOA_likelihood_list.append(tmp_TDOA_likelihoold)
    #                 avg_TDOA_likelihood = np.mean(np.array(TDOA_likelihood_list), axis=0)
    #                 # avg_TDOA_likelihood_list.append(avg_TDOA_likelihood)
    #
    #                 if avg_TDOA_likelihood > self.DOA_likelihood_threshold:
    #                     child_angle_list = []
    #                     for child_index in tmp_angle.child_indice:
    #                         child_angle_list.append(self.steer_angle_list[tmp_angle.depth+1][child_index])
    #
    #                     while len(child_angle_list) > 0:
    #                         tmp_child_angle = child_angle_list.pop()
    #
    #                         child_angle_TDOA_range = tmp_child_angle.TDOA_range
    #
    #                         child_TDOA_likelihood_list = []
    #                         for idx_pair, (range_min, range_max) in enumerate(child_angle_TDOA_range):
    #                             tmp_child_TDOA_likelihoold = np.max(tmp_tdoa_pred[idx_pair, range_min: range_max + 1],
    #                                                           axis=-1)
    #                             child_TDOA_likelihood_list.append(tmp_child_TDOA_likelihoold)
    #                         avg_child_TDOA_likelihood = np.mean(np.array(child_TDOA_likelihood_list), axis=0)
    #
    #
    #                         if avg_child_TDOA_likelihood <= self.DOA_likelihood_threshold or tmp_child_angle.depth + 1 == len(
    #                                 self.steer_angle_list):
    #                             estimated_DOAs.append((tmp_child_angle, avg_child_TDOA_likelihood))
    #                         else:
    #                             for child_index in tmp_child_angle.child_indice:
    #                                 child_angle_list.append(self.steer_angle_list[tmp_child_angle.depth+1][child_index])
    #
    #
    #             print("Debugging")



        # # For all depths
        # for steer_angles in self.steer_angle_list:
        #
        #     avg_TDOA_list = []
        #     for tmp_angle in steer_angles:
        #         angle_TDOA_range = tmp_angle.TDOA_range
        #
        #         TDOA_list = []
        #         for idx_pair, (range_min, range_max) in enumerate(angle_TDOA_range):
        #             tmp_TDOA = np.max(tdoa_pred[:, idx_pair, :, range_min: range_max + 1], axis=-1)
        #             TDOA_list.append(tmp_TDOA)
        #         avg_TDOA = np.mean(np.array(TDOA_list), axis=0)
        #         avg_TDOA_list.append(avg_TDOA)
        #
        #     print("Debugging")

        ############ TODO: Implement the estimating part of DOAs; IK, 20220524

    def estimate_DOA_v2(self, tdoa_pred): # for testing
        # E.g., [128, 6, 10, 21] --> The shape of GT DOAs: [128, 10, 9 (DOAs) + 1]
        batch_size, num_mic_pairs, num_frames, delay_len = tdoa_pred.shape

        num_angles = len(self.pts_horizontal)

        # The output: DOA predictions (Shape: 128, 10, 9 + 1)
        # doas_pred = np.zeros((batch_size, num_frames, 10))
        DOA_pred_function = np.zeros((batch_size, num_frames, num_angles))

        # For all batches
        for batch_idx in range(batch_size):
            # For all frames
            for frame_idx in range(num_frames):
                # Empty list of estimated DOAs
                estimated_DOAs = []
                computed_DOAs_list = []
                # estimated_DOAs_coord = []

                tmp_tdoa_pred = tdoa_pred[batch_idx, :, frame_idx, :]

                # tmp_DOA_likelihood_function = np.zeros((num_angles))
                # For all angles
                for angle_idx in range(num_angles):
                    steer_angle = self.pts_horizontal[angle_idx]
                    steer_TDOA_filter = self.TDOA_filter[angle_idx]

                    DOA_pred_function[batch_idx, frame_idx, angle_idx] = np.sum(steer_TDOA_filter * tmp_tdoa_pred) / num_mic_pairs
                    # tmp_DOA_likelihood_function[angle_idx] = np.sum(steer_TDOA_filter * tmp_tdoa_pred) / num_mic_pairs
                # print("Debugging")

        return DOA_pred_function

    # def initialize_TDOA_ranges(self, mic_pair_pos, half_delay_len):
    #     # For each depth
    #     for angles in self.steer_angle_list:
    #         for angle in angles:
    #             angle.compute_TDOA_range(mic_pair_pos, half_delay_len)

    def initialize_TDOA_filter(self, mic_pair_pos, half_delay_len):
        delay_len = 2 * half_delay_len + 1
        num_mic_pairs = len(mic_pair_pos)
        num_angles = len(self.pts_horizontal)

        TDOA_filter = np.zeros((num_angles, num_mic_pairs, delay_len))
        TDOA_delay_dummy = np.zeros((num_mic_pairs, delay_len))
        for idx_delay in range(delay_len):
            TDOA_delay_dummy[:, idx_delay] = idx_delay

        for angle_idx in range(num_angles):
            steer_angle = self.pts_horizontal[angle_idx]
            diff_TDOA_pair = utils.compute_diff_TDOA_float(np.array([steer_angle]), np.array([0, 0, 0]),
                                                                 mic_pair_pos, half_delay_len,
                                                                sampling_rate=self.fs)

            diff_TDOA_pair_dummy = np.tile(np.transpose(diff_TDOA_pair), (1, delay_len))
            diff_TDOA_pair_dummy_dist = diff_TDOA_pair_dummy - TDOA_delay_dummy

            gauss_filter = np.exp(-diff_TDOA_pair_dummy_dist * diff_TDOA_pair_dummy_dist / self.sigma_gauss_dist)

            TDOA_filter[angle_idx] = gauss_filter

        self.TDOA_filter = TDOA_filter



    # def initialize_recursive_circle(self, subdiv):
    #     # For all depth ...
    #     for i in range(subdiv):
    #         # Initialize vertice for the next depth
    #         angles = []
    #         chil_idx = 0
    #         for tmp_angle in self.steer_angle_list[i]:
    #             steer_angle = tmp_angle.steer_angle
    #             for range_angle in tmp_angle.steer_range:
    #                 middle_angle = (steer_angle + range_angle) / 2.0    # Between Steering angle and Range angle
    #                 angles.append(Angle(middle_angle, i + 1, fs=self.fs))
    #                 tmp_angle.set_child(chil_idx)
    #                 chil_idx = chil_idx + 1
    #
    #         # Set the computed angles to the next depth
    #         self.steer_angle_list[i+1] = angles
