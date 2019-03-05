#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasping Policy for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np

# Autolab imports
from autolab_core import RigidTransform
import trimesh
from visualization import Visualizer3D as vis3d

# 106B lab imports
from lab2.metrics import (
    compute_force_closure, 
    compute_gravity_resistance, 
    compute_custom_metric
)
from lab2.utils import length, normalize

# YOUR CODE HERE
# probably don't need to change these (BUT confirm that they're correct)
MAX_HAND_DISTANCE = .04
MIN_HAND_DISTANCE = .01
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1

# TODO
OBJECT_MASS = {'gearbox': .25, 'nozzle': .25, 'pawn': .25}


class GraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name, mesh, T_obj_world, T_world_ar):
        """
        Parameters
        ----------
        n_vert : int
            We are sampling vertices on the surface of the object, and will use pairs of 
            these vertices as grasp candidates
        n_grasps : int
            how many grasps to sample.  Each grasp is a pair of vertices
        n_execute : int
            how many grasps to return in policy.action()
        n_facets : int
            how many facets should be used to approximate the friction cone between the 
            finger and the object
        metric_name : string
            name of one of the function in src/lab2/metrics/metrics.py
        """
        self.n_vert = n_vert
        self.n_grasps = n_grasps
        self.n_execute = n_execute
        self.n_facets = n_facets
        # This is a function, one of the functions in src/lab2/metrics/metrics.py
        self.metric = eval(metric_name)
        self.mesh = mesh
        self.T_obj_world = T_obj_world
        self.T_world_ar = T_world_ar

        self.ar_z = T_world_ar.position[2]

    def vertices_to_baxter_hand_pose(grasp_vertices, approach_direction):
        """
        takes the contacts positions in the object frame and returns the hand pose T_obj_gripper
        BE CAREFUL ABOUT THE FROM FRAME AND TO FRAME.  the RigidTransform class' frames are 
        weird.
        
        Parameters
        ----------
        grasp_vertices : 2x3 :obj:`numpy.ndarray`
            position of the fingers in object frame
        approach_direction : 3x' :obj:`numpy.ndarray`
            there are multiple grasps that go through contact1 and contact2.  This describes which 
            orientation the hand should be in

        Returns
        -------
        :obj:`autolab_core:RigidTransform` Hand pose in the object frame
        """
        # YOUR CODE HERE
        raise NotImplementedError

    def sample_grasps(self, vertices, normals, cos_thresh=-0.8, table_thresh=0.03):
        """
        Samples a bunch of candidate grasps.  You should randomly choose pairs of vertices and throw out
        pairs which are too big for the gripper, or too close too the table.  You should throw out vertices 
        which are lower than ~3cm of the table.  You may want to change this.  Returns the pairs of 
        grasp vertices and grasp normals (the normals at the grasp vertices)

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals
        T_ar_object : :obj:`autolab_core.RigidTransform`
            transform from the AR tag on the paper to the object

        Returns
        -------
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps vertices.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector and there are n_grasps of them, hence the shape n_graspsx2x3
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        """
        grasp_vertices = None
        grasp_normals = None
        # points, face_indices = trimesh.sample.sample_surface_even(self.mesh, 1000)
        # normals = mesh.face_normals[face_indices]
        grasp_vertices = np.zeros((self.n_grasps, 2, 3))
        grasp_normals = np.zeros((self.n_grasps, 2, 3))


        i = 0
        while i < self.n_grasps:
            first = np.random.randint(0,len(vertices))
            second = np.random.randint(0, len(vertices))
            first_vert, second_vert = vertices[first], vertices[second]
            first_norm, second_norm = normals[first], normals[second]

            first_vert_homo = np.array(list(first_vert) + [1])
            second_vert_homo = np.array(list(second_vert) + [1])


            first_vert_world = self.T_obj_world.inverse().matrix.dot(first_vert_homo).reshape(-1)
            second_vert_world = self.T_obj_world.inverse().matrix.dot(second_vert_homo).reshape(-1)


            # antipodality
            if np.inner(first_norm, second_norm) > cos_thresh:
                continue

            # within gripper distance
            if np.linalg.norm(first_vert - second_vert) > MAX_HAND_DISTANCE or np.linalg.norm(first_vert - second_vert) < MIN_HAND_DISTANCE:
                continue

            # closeness to table
            min_vertex_z = min(first_vert_world[2], second_vert_world[2])
            if abs(min_vertex_z - self.ar_z) < table_thresh:
                continue

            grasp_vert = np.array([[first_vert], 
                                   [second_vert]
                                  ]).reshape((2, 3))
            grasp_norm = np.array([[first_norm], 
                                   [second_norm]
                                  ]).reshape((2, 3))
            grasp_vertices[i] = grasp_vert
            grasp_normals[i] = grasp_norm
            i += 1

        return grasp_vertices, grasp_normals



    def score_grasps(self, grasp_vertices, grasp_normals, object_mass):
        """
        takes mesh and returns pairs of contacts and the quality of grasp between the contacts, NOT sorted by quality
        
        Parameters
        ----------
        grasp_vertices : n_graspsx2x3 :obj:`numpy.ndarray`
            grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        grasp_normals : mx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3

        Returns
        -------
        :obj:`list` of int
            grasp quality for each 
        """
        return [self.metric(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass) for i in range(len(grasp_vertices))]

    def vis(self, mesh, grasp_vertices, grasp_qualities):
        """
        Pass in any grasp and its associated grasp quality.  this function will plot
        each grasp on the object and plot the grasps as a bar between the points, with
        colored dots on the line endpoints representing the grasp quality associated
        with each grasp
        
        Parameters
        ----------
        mesh : :obj:`Trimesh`
        grasp_vertices : mx2x3 :obj:`numpy.ndarray`
            m grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, hence the shape mx2x3
        grasp_qualities : mx' :obj:`numpy.ndarray`
            vector of grasp qualities for each grasp
        """
        vis3d.mesh(mesh)

        dirs = normalize(grasp_vertices[:,0] - grasp_vertices[:,1], axis=1)
        midpoints = (grasp_vertices[:,0] + grasp_vertices[:,1]) / 2
        grasp_endpoints = np.zeros(grasp_vertices.shape)
        grasp_vertices[:,0] = midpoints + dirs*MAX_HAND_DISTANCE/2
        grasp_vertices[:,1] = midpoints - dirs*MAX_HAND_DISTANCE/2

        for grasp, quality in zip(grasp_vertices, grasp_qualities):
            color = [min(1, 2*(1-quality)), min(1, 2*quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)
        vis3d.show()

    def top_n_actions(self, mesh, obj_name, vis=True):
        """
        Takes in a mesh, samples a bunch of grasps on the mesh, evaluates them using the 
        metric given in the constructor, and returns the best grasps for the mesh.  SHOULD
        RETURN GRASPS IN ORDER OF THEIR GRASP QUALITY.

        You should try to use mesh.mass to get the mass of the object.  You should check the 
        output of this, because from this
        https://github.com/BerkeleyAutomation/trimesh/blob/master/trimesh/base.py#L2203
        it would appear that the mass is approximated using the volume of the object.  If it
        is not returning reasonable results, you can manually weight the objects and store 
        them in the dictionary at the top of the file.

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        vis : bool
            Whether or not to visualize the top grasps

        Returns
        -------
        :obj:`list` of :obj:`autolab_core.RigidTransform`
            the matrices T_grasp_world, which represents the hand poses of the baxter / sawyer
            which would result in the fingers being placed at the vertices of the best grasps
        """
        # Some objects have vertices in odd places, so you should sample evenly across 
        # the mesh to get nicer candidate grasp points using trimesh.sample.sample_surface_even()
        
        points, face_indices = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        normals = mesh.face_normals[face_indices]
        grasp_vertices, grasp_normals = self.sample_grasps(points, normals)
        scores = self.score_grasps(grasp_vertices, grasp_normals, mesh.mass)


        top_scores_idx = sorted(zip(scores, range(len(scores))), key=lambda x: x[0], reverse=True)
        top_idx = [i for s, i in top_k_scores_idx][:self.n_execute]
        top_grasp_vertices = [grasp_vertices[i] for i in top_idx]
        top_grasp_normals = [grasp_normals[i] for i in top_idx]

        for grasp_verts, grasp_norm in zip(top_grasp_vertices, top_grasp_normals):
            # approach 1: nullspace
            # approach 2: negative reciprocal for [x, y], choose z rotation to match the plane of the normals, depending on
            # from which side the arm is coming in
            # vertices_to_baxter_hand_pose(grasp_verts, approach_direction)