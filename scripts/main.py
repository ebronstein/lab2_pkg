#!/home/cc/ee106b/sp19/class/ee106b-aap/ee106b_sp19/ros_workspaces/lab2_ws/env/bin/python -W ignore::DeprecationWarning
"""
Starter script for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
import scipy
import sys
import argparse

# AutoLab imports
from autolab_core import RigidTransform
import trimesh

# 106B lab imports
from lab2.policies import GraspingPolicy


try:
    import rospy
    import tf
    from baxter_interface import gripper as baxter_gripper
    from path_planner import PathPlanner
    ros_enabled = True
except:
    print 'Couldn\'t import ROS.  I assume you\'re running this on your laptop'
    ros_enabled = False

def lookup_transform(to_frame, from_frame='base'):
    """
    Returns the AR tag position in world coordinates 

    Parameters
    ----------
    to_frame : string
        examples are: ar_marker_7, gearbox, pawn, ar_marker_3, etc
    from_frame : string
        lets be real, you're probably only going to use 'base'

    Returns
    -------
    :obj:`autolab_core.RigidTransform` AR tag position or object in world coordinates
    """
    if not ros_enabled:
        print 'I am the lookup transform function!  ' \
            + 'You\'re not using ROS, so I\'m returning the Identity Matrix.'
        return RigidTransform(to_frame=from_frame, from_frame=to_frame)
    listener = tf.TransformListener()
    attempts, max_attempts, rate = 0, 10, rospy.Rate(1.0)

    # Sleep for a bit before looking up transformation
    for _ in range(1):
        rate.sleep()

    while attempts < max_attempts:
        try:
            # import pdb; pdb.set_trace()
            t = listener.getLatestCommonTime(from_frame, to_frame)
            tag_pos, tag_rot = listener.lookupTransform(from_frame, to_frame, t)
            break
        except:
            rate.sleep()
            attempts += 1
            print 'Attempt {0} to look up transformation from {1} to {2} unsuccessful.'.format(
                    attempts, from_frame, to_frame)
    print 'Successfully found transformation from {0} to {1}.'.format(from_frame, to_frame)
    # import pdb; pdb.set_trace()
    rot = RigidTransform.rotation_from_quaternion(tag_rot)
    return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)

def execute_grasp(T_grasp_world, planner, gripper):
    """
    takes in the desired hand position relative to the object, finds the desired 
    hand position in world coordinates.  Then moves the gripper from its starting 
    orientation to some distance BEHIND the object, then move to the  hand pose 
    in world coordinates, closes the gripper, then moves up.  
    
    Parameters
    ----------
    T_grasp_world : :obj:`autolab_core.RigidTransform`
        desired position of gripper relative to the world frame
    """
    def close_gripper():
        """closes the gripper"""
        gripper.close(block=True)
        rospy.sleep(1.0)

    def open_gripper():
        """opens the gripper"""
        gripper.open(block=True)
        rospy.sleep(1.0)

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return
    raise NotImplementedError

def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='gearbox', help=
        """Which Object you\'re trying to pick up.  Options: gearbox, nozzle, pawn.  
        Default: gearbox"""
    )
    parser.add_argument('-n_vert', type=int, default=1000, help=
        'How many vertices you want to sample on the object surface.  Default: 1000'
    )
    parser.add_argument('-n_facets', type=int, default=32, help=
        """You will approximate the friction cone as a set of n_facets vectors along 
        the surface.  This way, to check if a vector is within the friction cone, all 
        you have to do is check if that vector can be represented by a POSITIVE 
        linear combination of the n_facets vectors.  Default: 32"""
    )
    parser.add_argument('-n_grasps', type=int, default=500, help=
        'How many grasps you want to sample.  Default: 500')
    parser.add_argument('-n_execute', type=int, default=5, help=
        'How many grasps you want to execute.  Default: 5')
    parser.add_argument('-metric', '-m', type=str, default='compute_force_closure', help=
        """Which grasp metric in grasp_metrics.py to use.  
        Options: compute_force_closure, compute_gravity_resistance, compute_custom_metric"""
    )
    parser.add_argument('-arm', '-a', type=str, default='left', help=
        'Options: left, right.  Default: left'
    )
    parser.add_argument('--baxter', action='store_true', help=
        """If you don\'t use this flag, you will only visualize the grasps.  This is 
        so you can run this on your laptop"""
    )
    parser.add_argument('--debug', action='store_true', help=
        'Whether or not to use a random seed'
    )
    parser.add_argument('--ros', action='store_true', help=
        'Whether or not to use ROS'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        np.random.seed(0)

    ros_enabled = ros_enabled and args.ros

    rospy.init_node('lab2_node')

    # Mesh loading and pre-processing
    print 'Loading mesh'
    mesh = trimesh.load_mesh('objects/{}.obj'.format(args.obj))
    print 'Looking up transformation to the object.'
    T_obj_world = lookup_transform(args.obj)
    T_world_ar = None # TODO
    mesh.apply_transform(T_obj_world.matrix)
    mesh.fix_normals()

    print 'Initializing grasping policy.'

    # This policy takes a mesh and returns the best actions to execute on the robot
    grasping_policy = GraspingPolicy(
        args.n_vert, 
        args.n_grasps, 
        args.n_execute, 
        args.n_facets, 
        args.metric,
        mesh,
        T_obj_world,
        T_world_ar
    )

    print 'Computing grasps.'

    # Each grasp is represented by T_grasp_world, a RigidTransform defining the 
    # position of the end effector
    T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)

    # Execute each grasp on the baxter / sawyer
    if args.baxter:
        gripper = baxter_gripper.Gripper(args.arm)
        planner = PathPlanner('{}_arm'.format(arm))

        for T_grasp_world in T_grasp_worlds:
            repeat = True
            while repeat:
                execute_grasp(T_grasp_world, planner, gripper)
                repeat = raw_input("repeat? [y|n] ") == 'y'
