#!/home/cc/ee106b/sp19/class/ee106b-aap/ee106b_sp19/ros_workspaces/lab2_ws/env/bin/python -W ignore::DeprecationWarning
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
from lab2.utils import length, normalize, rotation2d

# YOUR CODE HERE
# probably don't need to change these (BUT confirm that they're correct)
MAX_HAND_DISTANCE = .04
MIN_HAND_DISTANCE = .01
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1

# TODO
OBJECT_MASS = {'gearbox': .25, 'nozzle': .25, 'pawn': .25}


class GraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name, mesh, T_obj_world, T_world_ar, object_name):
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
        self.object_name = object_name

        self.ar_z = T_world_ar.position[2] # 0.1

    # def vertices_to_baxter_hand_pose(grasp_vertices, approach_direction):
    #     """
    #     takes the contacts positions in the object frame and returns the hand pose T_obj_gripper
    #     BE CAREFUL ABOUT THE FROM FRAME AND TO FRAME.  the RigidTransform class' frames are 
    #     weird.
        
    #     Parameters
    #     ----------
    #     grasp_vertices : 2x3 :obj:`numpy.ndarray`
    #         position of the fingers in object frame
    #     approach_direction : 3x' :obj:`numpy.ndarray`
    #         there are multiple grasps that go through contact1 and contact2.  This describes which 
    #         orientation the hand should be in

    #     Returns
    #     -------
    #     :obj:`autolab_core:RigidTransform` Hand pose in the object frame
    #     """
    #     # YOUR CODE HERE
    
    def vertices_to_baxter_hand_pose(self, grasp_vertices, grasp_normals):
        """
        takes the contacts positions in the object frame and returns the hand pose T_obj_gripper
        BE CAREFUL ABOUT THE FROM FRAME AND TO FRAME.  the RigidTransform class' frames are 
        weird.
        
        Parameters
        ----------
        grasp_vertices : 2x3 :obj:`numpy.ndarray`
            position of the fingers in object frame
        grasp_normals : 2x3' :obj:`numpy.ndarray`
            normals at the grasp vertices

        Returns
        -------
        :obj:`autolab_core:RigidTransform` Hand pose in the object frame
        """
        # YOUR CODE HERE
        # y-axis (bar across the grippers): direction of the grasp normals

        y_axis = (grasp_normals[0] - grasp_normals[1]) / 2
        y_axis /= np.linalg.norm(y_axis)

        # z-axis (direction the end-effector is pointing, pointing out of the grippers): take the first two terms of the y-axis 
        # (in the world frame, or in object frame if the object doesn't have rotation) 
        # (think of it in 2D),
        # rotate it by pi/2, and then set the third term to 0 because we want
        # to have a "flat" z-axis in the world frame
        y_axis_2d = y_axis[:2]
        z_axis_2d = rotation2d(np.pi / 2).dot(y_axis_2d)
        z_axis = np.hstack([z_axis_2d, [0]])
        z_axis /= np.linalg.norm(z_axis)

        # x-axis: take the cross product of the y- and z-axes
        x_axis = np.cross(y_axis, z_axis)

        rotation_3d = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
        translation = np.average(grasp_vertices, axis=0) # middle position between the grasp vertices
        return RigidTransform(rotation=rotation_3d, translation=translation, to_frame=self.object_name, from_frame='gripper')


    def sample_grasps(self, vertices, normals, normals_cos_thresh=-0.9, contact_alignment_cos_threshold=0.9, table_thresh=0.03):
        """
        Samples a bunch of candidate grasps.  You should randomly choose pairs of vertices and throw out
        pairs which are too big for the gripper, or too close too the table.  You should throw out vertices 
        which are lower than ~3cm of the table.  You may want to change this.  Returns the pairs of 
        grasp vertices and grasp normals (the normals at the grasp vertices)

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices (frame: object)
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals (frame: object)
        T_ar_object : :obj:`autolab_core.RigidTransform`
            transform from the AR tag on the paper to the object

        Returns
        -------
        n_graspsx2x3 :obj:`numpy.ndarray` (frame: object)
            grasps vertices.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector and there are n_grasps of them, hence the shape n_graspsx2x3
        n_graspsx2x3 :obj:`numpy.ndarray` (frame: object)
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        """
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
            # normals should be aligned
            if np.inner(first_norm, second_norm) > normals_cos_thresh:
                # print 'Rejecting grasp due to normals not being aligned.'
                continue
            # points should be aligned with gripper
            v12 = second_vert - first_vert
            if np.inner(v12, second_norm) / np.linalg.norm(v12) < contact_alignment_cos_threshold:
                # print 'Rejecting grasp due to contacts not being aligned for the gripper.'
                continue
            v21 = first_vert - second_vert
            if np.inner(v21, first_norm) / np.linalg.norm(v21) < contact_alignment_cos_threshold:
                # print 'Rejecting grasp due to contacts not being aligned for the gripper.'
                continue

            # correct position (normals should face away from each other)
            dist = np.linalg.norm(first_vert - second_vert)
            dist_along_norm = np.linalg.norm(first_vert + 1e-6 * first_norm - second_vert)
            dist_against_norm = np.linalg.norm(first_vert - 1e-6 * first_norm - second_vert)
            if dist_along_norm < dist or dist_along_norm < dist_against_norm:
                print 'Rejecting grasp due to normals facing toward each other'
                continue

            # within gripper distance
            if np.linalg.norm(first_vert - second_vert) > MAX_HAND_DISTANCE or np.linalg.norm(first_vert - second_vert) < MIN_HAND_DISTANCE:
                continue

            # closeness to table
            min_vertex_z = min(first_vert_world[2], second_vert_world[2])
            if abs(min_vertex_z - self.ar_z) < table_thresh:
                print 'Rejecting grasp due to closeness to table'
                continue

            grasp_vert = np.array([[first_vert], 
                                   [second_vert]
                                  ]).reshape((2, 3))
            grasp_norm = np.array([[first_norm], 
                                   [second_norm]
                                  ]).reshape((2, 3))
            grasp_vertices[i] = grasp_vert
            grasp_normals[i] = grasp_norm

            # if i % 10 == 0:
            print i
            i += 1

        return grasp_vertices, grasp_normals # (frame: object)



    def score_grasps(self, grasp_vertices, grasp_normals, object_mass):
        """
        takes mesh and returns pairs of contacts and the quality of grasp between the contacts, NOT sorted by quality
        
        Parameters
        ----------
        grasp_vertices : n_graspsx2x3 :obj:`numpy.ndarray` (frame: object)
            grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        grasp_normals : mx2x3 :obj:`numpy.ndarray` (frame: object)
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3

        Returns
        -------
        :obj:`list` of int
            grasp quality for each 
        """
        return [self.metric(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass) for i in range(len(grasp_vertices))]

    def vis(self, mesh, grasp_vertices, grasp_normals, grasp_qualities, hand_poses):
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
        hand_poses: list of RigidTransform objects
        """
        vis3d.mesh(mesh)

        dirs = normalize(grasp_vertices[:,0] - grasp_vertices[:,1], axis=1)
        midpoints = (grasp_vertices[:,0] + grasp_vertices[:,1]) / 2
        grasp_contacts = np.zeros(grasp_vertices.shape)
        grasp_contacts[:,0] = midpoints + dirs*MAX_HAND_DISTANCE/2
        grasp_contacts[:,1] = midpoints - dirs*MAX_HAND_DISTANCE/2

        for grasp, quality, normal, contacts, hand_pose in zip(grasp_vertices, grasp_qualities, grasp_normals, grasp_contacts, hand_poses):
            color = [min(1, 2*(1-quality)), min(1, 2*quality), 0, 1]

            vis3d.points(grasp, scale=0.001, color=(1,0,0))
            n1_endpoints = np.zeros((2,3))
            n1_endpoints[0] = grasp[0]
            n1_endpoints[1] = grasp[0] + 0.01 * normal[0]

            n2_endpoints = np.zeros((2,3))
            n2_endpoints[0] = grasp[1]
            n2_endpoints[1] = grasp[1] + 0.01 * normal[1]

            vis3d.plot3d(n1_endpoints, color=color, tube_radius=0.001)
            vis3d.plot3d(n2_endpoints, color=color, tube_radius=0.001)

            # hand pose
            vis3d.points(hand_pose.position, scale=0.001, color=(0.5, 0.5, 0.5))

            x_axis_endpoints = np.zeros((2, 3))
            x_axis_endpoints[0] = hand_pose.position
            x_axis_endpoints[1] = hand_pose.position + 0.01 * hand_pose.x_axis
            vis3d.plot3d(x_axis_endpoints, color=(1,0,0), tube_radius=0.0001)
            
            y_axis_endpoints = np.zeros((2, 3))
            y_axis_endpoints[0] = hand_pose.position
            y_axis_endpoints[1] = hand_pose.position + 0.01 * hand_pose.y_axis
            vis3d.plot3d(y_axis_endpoints, color=(0,1,0), tube_radius=0.0001)
            
            z_axis_endpoints = np.zeros((2, 3))
            z_axis_endpoints[0] = hand_pose.position
            z_axis_endpoints[1] = hand_pose.position + 0.01 * hand_pose.z_axis
            vis3d.plot3d(z_axis_endpoints, color=(0,0,1), tube_radius=0.0001)

            vis3d.plot3d(contacts, color=(0, 0, 1), tube_radius=.0005)
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
        
        points, face_indices = trimesh.sample.sample_surface_even(mesh, self.n_vert) # (frame: object)
        normals = mesh.face_normals[face_indices] # (frame: object)
        grasp_vertices, grasp_normals = self.sample_grasps(points, normals) # (frame: object)
        scores = self.score_grasps(grasp_vertices, grasp_normals, mesh.mass)


        top_k_scores_idx = sorted(zip(scores, range(len(scores))), key=lambda x: x[0], reverse=True)
        top_grasp_qualities = [s for s, i in top_k_scores_idx][:self.n_execute] 
        top_idx = [i for s, i in top_k_scores_idx][:self.n_execute]
        top_grasp_vertices = np.array([grasp_vertices[i] for i in top_idx])
        top_grasp_normals = np.array([grasp_normals[i] for i in top_idx])

        # Get the hand poses
        poses = []
        for grasp_verts, grasp_norm in zip(top_grasp_vertices, top_grasp_normals):
            poses.append(self.vertices_to_baxter_hand_pose(grasp_verts, grasp_norm))

        # Visualize the grasps
        if vis:
            self.vis(mesh, top_grasp_vertices, top_grasp_normals, top_grasp_qualities, poses)
        

        # tf_inv = self.T_obj_world.inverse()
        # our_world_poses = [tf_inv.matrix.dot(pose.matrix) for pose in poses]
        # world_poses = [pose * tf_inv for pose in poses]
        
        world_poses = [self.T_obj_world * pose for pose in poses]
        import pdb; pdb.set_trace()
        return world_poses