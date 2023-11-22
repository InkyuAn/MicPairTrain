import sys
import os

import trimesh
from trimesh import creation

import scipy.io

import random
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import mpl_toolkits.mplot3d

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# import get_param as parameters

from utils.sphere_grid import Icosphere


def main(ico_depth):
    print("Generate and Save Icosphere grids ... depth: ", ico_depth)
    # Get parameters
    # project_dir = '/root/Projects/DL_based_SSL_project/DL_based_SSL'

    # saving_dir = os.path.join(project_dir, 'utils')

    # saving_np_dir = os.path.join(saving_dir, 'icosphere_vertices_d%d' % ico_dpeth)
    # saving_mat_dir = os.path.join(saving_dir, 'icosphere_vertices_d%d.mat' % ico_dpeth)

    # icosphere = trimesh.creation.icosphere(ico_dpeth)
    # icosphere_vertices = np.array(icosphere.vertices)

    # np.save(saving_np_dir, icosphere_vertices)
    # scipy.io.savemat(saving_mat_dir, {'icosphere_vertices': icosphere_vertices})

    # icosphere, face = utils.icosphere.icosphere(ico_dpeth)
    # icosphere_vertices = np.array(icosphere.vertices)

    icosphere_node = Icosphere(ico_depth)

    ############################################################################
    ### Steering
    steer_neigh_list = [[] for i in range(ico_depth + 1)]
    child_point_list = []
    for idx in range(len(icosphere_node.steer_vertice_list[0])):
        child_point_list.append(idx)

    for tmp_depth in range(ico_depth + 1):
        vertice = icosphere_node.steer_vertice_list[tmp_depth]

        steering_vertex_idx = random.choice(child_point_list)
        # For debugging
        if tmp_depth == 0:
            steering_vertex_idx = 8

        neigh_indice = vertice[steering_vertex_idx].neighbor_indice

        steer_neigh_list[tmp_depth].append(steering_vertex_idx)
        for neigh_idx in neigh_indice:
            steer_neigh_list[tmp_depth].append(neigh_idx)

        print("[Depth ", tmp_depth, "]")
        print("  Child, ", child_point_list)
        print("  .. Steering idx, ", steering_vertex_idx)
        print("  .. Neigh, ", neigh_indice)
        print("  .. Neigh, ", steer_neigh_list[tmp_depth])



        child_vertice = vertice[steering_vertex_idx].child_indice
        child_point_list.clear()
        for child_vertex in child_vertice:
            child_point_list.append(child_vertex)
    ############################################################################
    print("**************************")


    fig = plt.figure()
    for tmp_depth in range(ico_depth + 1):

        vertices = icosphere_node.steer_vertice_list[tmp_depth]
        faces = icosphere_node.steer_faces_list[tmp_depth]
        faces_np = np.array(faces)
        vertices_np = np.array([vertex.get_pos() for vertex in vertices])

        # basic mesh color, divided in 20 groups (one for each original face)
        jet = matplotlib.cm.tab20(np.linspace(0,1,20))
        jet = np.tile(jet[:,:3], (1, faces_np.shape[0]//20))
        jet = jet.reshape(faces_np.shape[0], 1, 3)

        # computing face shading intensity based on face normals
        face_normals = np.cross(vertices_np[faces_np[:,1]]-vertices_np[faces_np[:,0]],
                                vertices_np[faces_np[:,2]]-vertices_np[faces_np[:,0]])

        face_normals /= np.sqrt(np.sum(face_normals**2, axis=1, keepdims=True))
        light_source = matplotlib.colors.LightSource(azdeg=60, altdeg=30)
        intensity = light_source.shade_normals(face_normals)

        # blending face colors and face shading intensity
        rgb = light_source.blend_hsv(rgb=jet, intensity=intensity.reshape(-1,1,1))

        # adding alpha value, may be left out
        # rgba = np.concatenate((rgb, 0.9*np.ones(shape=(rgb.shape[0],1,1))), axis=2)
        rgba = np.concatenate((rgb, 0.3*np.ones(shape=(rgb.shape[0],1,1))), axis=2)

        # creating mesh with given face colors
        poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(vertices_np[faces_np])
        poly.set_facecolor(rgba.reshape(-1,4))
        poly.set_edgecolor('black')
        poly.set_linewidth(0.25)

        # and now -- visualization!
        ax = fig.add_subplot(2,3,tmp_depth + 1, projection='3d')
        ax.add_collection3d(poly)

        # Add steering points

        # Extract steering points
        # if tmp_depth == 1:
        #     # steer_neigh = [8, 31, 30, 40, 39, 41] # Children of 8
        #     steer_neigh = [7, 17, 8, 30, 39, 28]
        #     steer_neigh_points = vertices_np[steer_neigh]
        # else:
        #     steer_neigh = steer_neigh_list[tmp_depth]
        #     steer_neigh_points = vertices_np[steer_neigh]

        steer_neigh = steer_neigh_list[tmp_depth]
        steer_neigh_points = vertices_np[steer_neigh]

        ax.scatter(steer_neigh_points[:, 0], steer_neigh_points[:, 1], steer_neigh_points[:, 2], color='r')
        # print("Points: ")
        # for steering_points in steer_neigh_points:
        #     print("(", steering_points, ")", end=" ")

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)

        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([-1,0,1])

        ax.set_title(f'nu={tmp_depth + 1}')
    fig.suptitle('Icospheres with different subdivision frequency')

    plt.show()
    # print("Icosphere vertices are saved ... ", icosphere_vertices.shape)
    # print("Test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute generate_icosphere_vertices')

    parser.add_argument('--ico_depth', metavar='ico_depth', type=int, default=3)
    args = parser.parse_args()

    try:
        # sys.exit(main(args.use_sslr, args.use_dcase, args.epoch, args.batch, args.model_version))
        sys.exit(main(args.ico_depth))
    except (ValueError, IOError) as e:
        sys.exit(e)