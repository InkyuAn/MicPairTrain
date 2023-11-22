import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import utils.utils as utils

# def vertex(x, y, z):
#     """ Return vertex coordinates fixed to the unit sphere """
#     length = np.sqrt(x**2 + y**2 + z**2)
#     return [i / length for i in (x,y,z)]
#
# def middle_point(verts,middle_point_cache,point_1, point_2):
#     """ Find a middle point and project to the unit sphere """
#     # We check if we have already cut this edge first
#     # to avoid duplicated verts
#     smaller_index = min(point_1, point_2)
#     greater_index = max(point_1, point_2)
#     key = '{0}-{1}'.format(smaller_index, greater_index)
#     if key in middle_point_cache: return middle_point_cache[key]
#     # If it's not in cache, then we can cut it
#     vert_1 = verts[point_1]
#     vert_2 = verts[point_2]
#     middle = [sum(i)/2 for i in zip(vert_1, vert_2)]
#     verts.append(vertex(*middle))
#     index = len(verts) - 1
#     middle_point_cache[key] = index
#     return index
#
# def icosphere(subdiv):
#     # verts for icosahedron
#     r = (1.0 + np.sqrt(5.0)) / 2.0
#     verts = np.array([[-1.0, r, 0.0],[ 1.0, r, 0.0],[-1.0, -r, 0.0],
#                       [1.0, -r, 0.0],[0.0, -1.0, r],[0.0, 1.0, r],
#                       [0.0, -1.0, -r],[0.0, 1.0, -r],[r, 0.0, -1.0],
#                       [r, 0.0, 1.0],[ -r, 0.0, -1.0],[-r, 0.0, 1.0]])
#     # rescale the size to radius of 0.5
#     verts /= np.linalg.norm(verts[0])
#     # adjust the orientation
#     r = R.from_quat([[0.19322862,-0.68019314,-0.19322862,0.68019314]])
#     verts = r.apply(verts)
#     verts = list(verts)
#
#     faces = [[0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],
#              [0, 10, 11],[1, 5, 9],[5, 11, 4],[11, 10, 2],
#              [10, 7, 6],[7, 1, 8],[3, 9, 4],[3, 4, 2],
#              [3, 2, 6],[3, 6, 8],[3, 8, 9],[5, 4, 9],
#              [2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1],];
#
#     for i in range(subdiv):
#         middle_point_cache = {}
#         faces_subdiv = []
#         for tri in faces:
#             v1  = middle_point(verts,middle_point_cache,tri[0], tri[1])
#             v2  = middle_point(verts,middle_point_cache,tri[1], tri[2])
#             v3  = middle_point(verts,middle_point_cache,tri[2], tri[0])
#             faces_subdiv.append([tri[0], v1, v3])
#             faces_subdiv.append([tri[1], v2, v1])
#             faces_subdiv.append([tri[2], v3, v2])
#             faces_subdiv.append([v1, v2, v3])
#         faces = faces_subdiv
#
#     return np.array(verts), np.array(faces)

### Vertex class
class Vertex:
    def __init__(self, x, y, z, fs=24000):
        self.fs = fs
        self.pos = self.normalizing(x, y, z)
        self.child_indice = []
        self.neighbor_indice = []
        self.TDOA_range = [] # The list of min % max of TDOA delays for all pairs

    def normalizing(self, x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """
        length = np.sqrt(x**2 + y**2 + z**2)
        return [i / length for i in (x,y,z)]

    def set_child(self, input_idx):
        if input_idx not in self.child_indice:
            self.child_indice.append(input_idx)

    def get_pos(self):
        return np.array(self.pos)

    def compute_TDOA_range(self, vertices_list, mic_pair_pos, half_delay_len):
        # Extract neighbor vertices
        vertex_neigh_pos_list = [vertices_list[veigh_idx].get_pos() for veigh_idx in self.neighbor_indice]

        # Compute ranges of TDOAs of neighbors
        vertex_neigh_diff_list = utils.utils.compute_diff_TDOA(vertex_neigh_pos_list, np.array([0, 0, 0]),
                                                                        mic_pair_pos, half_delay_len,
                                                               sampling_rate=self.fs)
        self.TDOA_range = np.stack((np.min(vertex_neigh_diff_list, axis=0), np.max(vertex_neigh_diff_list, axis=0)),
                                   axis=-1).tolist()

### Icosphere class
class Icosphere:
    # DOA_likelihood_threshold = 0.4 , prior value  1
    # DOA_likelihood_threshold = 0.3 , prior value  2
    # DOA_likelihood_threshold = 0.25 , prior value  3
    def __init__(self, subdiv, fs=24000, DOA_likelihood_threshold=0.3):
        self.fs = fs
        self.DOA_likelihood_threshold = DOA_likelihood_threshold
        # self.sigma_gauss_dist = 1.5   # Prior value   1
        # self.sigma_gauss_dist = 3.0   # Prior value   2
        # self.sigma_gauss_dist = 3.5   # Prior value   3
        self.sigma_gauss_dist = 3.5
        self.TDOA_filter = None

        # Generate Icosahedron consisting of 20 faces having the same area
        r = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([[-1.0, r, 0.0],[ 1.0, r, 0.0],[-1.0, -r, 0.0],
                          [1.0, -r, 0.0],[0.0, -1.0, r],[0.0, 1.0, r],
                          [0.0, -1.0, -r],[0.0, 1.0, -r],[r, 0.0, -1.0],
                          [r, 0.0, 1.0],[ -r, 0.0, -1.0],[-r, 0.0, 1.0]])
        # rescale the size to radius of 0.5
        verts /= np.linalg.norm(verts[0])
        # adjust the orientation
        r = R.from_quat([[0.19322862,-0.68019314,-0.19322862,0.68019314]])
        verts = r.apply(verts)
        verts = list(verts)

        faces = [[0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],
                 [0, 10, 11],[1, 5, 9],[5, 11, 4],[11, 10, 2],
                 [10, 7, 6],[7, 1, 8],[3, 9, 4],[3, 4, 2],
                 [3, 2, 6],[3, 6, 8],[3, 8, 9],[5, 4, 9],
                 [2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1],]

        # Convert Geometry information of Icosahedron to "Vertex" data structure
        # , which has the child and neighbor information

        # vertice = []
        # for vert in verts:
            # vertice.append(Vertex(vert[0], vert[1], vert[2], fs=self.fs))

        # Compute neighbors of each vertex
        # self.compute_neighbor_vertices(vertice, faces)

        # Initialize the empty lists for vertices and faces for each depth
        self.steer_vertice_list = [[] for i in range(subdiv + 1)]
        self.steer_faces_list = [[] for i in range(subdiv + 1)]

        # self.steer_vertice_list[0] = vertice
        self.steer_vertice_list[0] = verts
        self.steer_faces_list[0] = faces

        # Perform dividing each face into four faces, and generate data structure for Icosphere
        self.initialize_recursive_icosphere(subdiv)

    def initialize_TDOA_filter(self, mic_pair_pos, half_delay_len):
        angles = self.steer_vertice_list[-1]

        delay_len = 2 * half_delay_len + 1
        num_mic_pairs = len(mic_pair_pos)
        num_angles = len(angles)

        TDOA_filter = np.zeros((num_angles, num_mic_pairs, delay_len))
        TDOA_delay_dummy = np.zeros((num_mic_pairs, delay_len))
        for idx_delay in range(delay_len):
            TDOA_delay_dummy[:, idx_delay] = idx_delay

        for angle_idx in range(num_angles):
            steer_angle = angles[angle_idx]
            diff_TDOA_pair = utils.utils.compute_diff_TDOA_float(np.array([steer_angle]), np.array([0, 0, 0]),
                                                                 mic_pair_pos, half_delay_len,
                                                                sampling_rate=self.fs)

            diff_TDOA_pair_dummy = np.tile(np.transpose(diff_TDOA_pair), (1, delay_len))
            diff_TDOA_pair_dummy_dist = diff_TDOA_pair_dummy - TDOA_delay_dummy

            gauss_filter = np.exp(-diff_TDOA_pair_dummy_dist * diff_TDOA_pair_dummy_dist / self.sigma_gauss_dist)

            TDOA_filter[angle_idx] = gauss_filter

        self.TDOA_filter = TDOA_filter

    def estimate_DOA_v2(self, tdoa_pred, num_DOA_label, gts, grid_on_hemi=False):
        # E.g., [128, 6, 18, 10, 21] --> The shape of GT DOAs: [128, 10, 30 (DOAs of labels)]
        # DOAs of labels: Index of x (-> label_idx), y (-> label_idx + num_label*1), and z (-> label_idx + num_label*2)
        batch_size, num_mic_pairs, num_labels, num_frames, delay_len = tdoa_pred.shape

        angles = self.steer_vertice_list[-1]
        num_angles = len(angles)

        # The output: DOA predictions (Shape: 128, 10, 10 * 3)
        doas_pred = np.zeros((batch_size, num_frames, num_DOA_label * 3))

        # for all batches
        for batch_idx in range(batch_size):
            # For all frames
            for frame_idx in range(num_frames):
                for label_idx in range(num_DOA_label):
                    # # Empty list of estimated DOAs
                    # estimated_DOAs = []

                    tmp_gt = np.array([
                        gts[batch_idx, frame_idx, label_idx],
                        gts[batch_idx, frame_idx, label_idx + num_DOA_label],
                        gts[batch_idx, frame_idx, label_idx + num_DOA_label * 2]
                    ])
                    tmp_tdoa_pred = tdoa_pred[batch_idx, :, label_idx, frame_idx, :]
                    tmp_DOA_likelihood_function = np.zeros((num_angles))

                    # For all angles
                    for angle_idx in range(num_angles):
                        steer_TDOA_filter = self.TDOA_filter[angle_idx]
                        tmp_DOA_likelihood_function[angle_idx] = np.sum(
                            steer_TDOA_filter * tmp_tdoa_pred) / num_mic_pairs

                    if np.max(tmp_DOA_likelihood_function) > self.DOA_likelihood_threshold:
                        max_angle_idx = np.argmax(tmp_DOA_likelihood_function)
                        max_angle = angles[max_angle_idx]

                        doas_pred[batch_idx, frame_idx, label_idx] = max_angle[0]
                        doas_pred[batch_idx, frame_idx, label_idx + num_DOA_label] = max_angle[1]
                        if grid_on_hemi:
                            doas_pred[batch_idx, frame_idx, label_idx + num_DOA_label * 2] = np.abs(max_angle[2])
                        else:
                            doas_pred[batch_idx, frame_idx, label_idx + num_DOA_label * 2] = max_angle[2]
                    # print("Debugging")
        return doas_pred

    def estimate_DOA(self, tdoa_pred, num_DOA_label):
        # E.g., [128, 6, 18, 10, 21] --> The shape of GT DOAs: [128, 10, 30 (DOAs of labels)]
        # DOAs of labels: Index of x (-> label_idx), y (-> label_idx + num_label*1), and z (-> label_idx + num_label*2)
        batch_size, num_mic_pairs, num_labels, num_frames, delay_len = tdoa_pred.shape

        # The output: DOA predictions (Shape: 128, 10, 10 * 3)
        doas_pred = np.zeros((batch_size, num_frames, num_DOA_label * 3))

        # for all batches
        for batch_idx in range(batch_size):
            # For all frames
            for frame_idx in range(num_frames):
                for label_idx in range(num_DOA_label):
                    # Empty list of estimated DOAs
                    estimated_DOAs = []

                    tmp_tdoa_pred = tdoa_pred[batch_idx, :, label_idx, frame_idx, :]

                    # For the deepest depth
                    steer_vetices = self.steer_vertice_list[-1]

                    for idx_vertex, tmp_vetex in enumerate(steer_vetices):
                        # Extract the TDOA range of the steering vertex
                        angle_TDOA_range = tmp_vetex.TDOA_range

                        # Compute the DOA likelihood
                        DOA_likelihood_list = []
                        for idx_pair, (range_min, range_max) in enumerate(angle_TDOA_range):
                            tmp_TDOA_likelihoold = np.max(tmp_tdoa_pred[idx_pair, range_min: range_max + 1], axis=-1)
                            DOA_likelihood_list.append(tmp_TDOA_likelihoold)
                        # Compute the mean of the DOA likelihood of all mic's pairs
                        avg_DOA_likelihood = np.mean(np.array(DOA_likelihood_list), axis=0)

                        # If the DOA likelihood is larger than the DOA likelihood,
                        if avg_DOA_likelihood > self.DOA_likelihood_threshold:
                            estimated_DOAs.append((avg_DOA_likelihood, idx_vertex, tmp_vetex))

                    # If there exist esimated DOAs,
                    if len(estimated_DOAs) > 0:
                        # Need to do something ...
                        print("Check here...")
        print("Debugging")


    def initialize_TDOA_ranges(self, mic_pair_pos, half_delay_len):
        # For each depth
        for vertices in self.steer_vertice_list:
            for vertex in vertices:
                vertex.compute_TDOA_range(vertices, mic_pair_pos, half_delay_len)

    def initialize_recursive_icosphere(self, subdiv):
        # For all depths ...
        for i in range(subdiv):
            # Initilize vertice for the next depth
            verts = []
            for idx, tmp_vert in enumerate(self.steer_vertice_list[i]):
                # tmp_vert.set_child(idx)
                # verts.append(Vertex(tmp_vert.pos[0], tmp_vert.pos[1], tmp_vert.pos[2], self.fs))
                verts.append(tmp_vert)

            middle_point_cache = {}
            faces_subdiv = []
            for tri in self.steer_faces_list[i]:
                # Compute the middle points of each edge of the face
                v1  = self.middle_point(verts, middle_point_cache, tri[0], tri[1])
                v2  = self.middle_point(verts, middle_point_cache, tri[1], tri[2])
                v3  = self.middle_point(verts, middle_point_cache, tri[2], tri[0])
                faces_subdiv.append([tri[0], v1, v3])
                faces_subdiv.append([tri[1], v2, v1])
                faces_subdiv.append([tri[2], v3, v2])
                faces_subdiv.append([v1, v2, v3])

                # # Set child: the generated middle points become children of the vertice of the upper depth
                # self.steer_vertice_list[i][tri[0]].set_child(v1)
                # self.steer_vertice_list[i][tri[0]].set_child(v3)
                # self.steer_vertice_list[i][tri[1]].set_child(v2)
                # self.steer_vertice_list[i][tri[1]].set_child(v1)
                # self.steer_vertice_list[i][tri[2]].set_child(v3)
                # self.steer_vertice_list[i][tri[2]].set_child(v2)

            # Compute neighbors of each vertex
            # self.compute_neighbor_vertices(verts, faces_subdiv)

            # Set the computed vertices and faces to the next depth
            self.steer_vertice_list[i+1] = verts
            self.steer_faces_list[i+1] = faces_subdiv

    def middle_point(self, verts, middle_point_cache, point_1, point_2):
        """ Find a middle point and project to the unit sphere """
        # We check if we have already cut this edge first
        # to avoid duplicated verts
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)
        key = '{0}-{1}'.format(smaller_index, greater_index)
        if key in middle_point_cache: return middle_point_cache[key]
        # If it's not in cache, then we can cut it
        # vert_1 = verts[point_1].pos
        # vert_2 = verts[point_2].pos
        vert_1 = verts[point_1]
        vert_2 = verts[point_2]
        middle = [sum(i)/2 for i in zip(vert_1, vert_2)]
        # verts.append(vertex(*middle))
        # verts.append(Vertex(*middle, fs=self.fs))
        verts.append(middle)
        index = len(verts) - 1
        middle_point_cache[key] = index
        return index

    def compute_neighbor_vertices(self, vertices, faces):

        num_vertices = len(vertices)
        neighbor_vertices = [[] for i in range(num_vertices)]
        # Check all vertices of each face
        for face in faces:


            neighbor_vertices[face[0]].append(face[1])
            neighbor_vertices[face[0]].append(face[2])

            neighbor_vertices[face[1]].append(face[0])
            neighbor_vertices[face[1]].append(face[2])

            neighbor_vertices[face[2]].append(face[0])
            neighbor_vertices[face[2]].append(face[1])

        # Refine the list of neighbor vertices; To remove repeated vertices
        for i in range(num_vertices):
            for vertex_idx in neighbor_vertices[i]:
                if vertex_idx not in vertices[i].neighbor_indice:
                    vertices[i].neighbor_indice.append(vertex_idx)


