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
from lab2.utils import create_pose_stamped_from_rigid_tf


try:
    import rospy
    import tf
    from baxter_interface import gripper as baxter_gripper
    from path_planner import PathPlanner
    from moveit_msgs.msg import DisplayTrajectory, RobotState
    from visualization_msgs.msg import Marker
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
            # original
            # tag_pos, tag_rot = listener.lookupTransform(from_frame, to_frame, t)
            # changes from piazza
            tag_pos, tag_rot = listener.lookupTransform(to_frame, from_frame, t)
            tag_rot = np.roll(tag_rot, 1)
            break
        except:
            rate.sleep()
            attempts += 1
            print 'Attempt {0} to look up transformation from {1} to {2} unsuccessful.'.format(
                    attempts, from_frame, to_frame)
    print 'Successfully found transformation from {0} to {1}.'.format(from_frame, to_frame)
    # import pdb; pdb.set_trace()
    rot = RigidTransform.rotation_from_quaternion(tag_rot)
    # original
    return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)
    # suggested by another group
    # return RigidTransform(rot, tag_pos, to_frame=to_frame, from_frame=from_frame)



def visualize_path(robot_trajectory, position, planned_path_pub, goal_pos_pub):
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory)
    disp_traj.trajectory_start = RobotState()
    planned_path_pub.publish(disp_traj)

    marker = Marker()
    marker.header.frame_id = "base"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    print 'Goal position {0}'.format(position)
    
    goal_pos_pub.publish(marker)

def plan_to_pose_custom(pose_stamped, planner):
    for i in range(100):
        plan = planner.plan_to_pose(pose_stamped)
        if len(plan.joint_trajectory.points) > 0:
            print 'Found plan!'
            break
        else:
            print 'Failed to find plan. Retrying...'
    return plan

def execute_grasp(T_grasp_world, planner, gripper, planned_path_pub, goal_pos_pub):
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

    inp = raw_input('Press <Enter> to move and open the gripper, or \'exit\' to exit')
    if inp == "exit":
        return
    else:
        open_gripper()

        # # dummy position ######################################################
        # quaternion = np.array([1, 0, 0, 0])
        # dummy_rotation = RigidTransform.rotation_from_quaternion(quaternion)
        # dummy_translation = np.array([0.1, 0.1, 0.1])
        # dummy_tf = RigidTransform(dummy_rotation, dummy_translation, from_frame='gripper', to_frame='base')
        # dummy_pose_stamped = create_pose_stamped_from_rigid_tf(dummy_tf)
        # dummy_pose_stamped.header.frame_id = 'base'
        # for i in range(10):
        #     dummy_plan = planner.plan_to_pose(dummy_pose_stamped)
        #     if len(dummy_plan.joint_trajectory.points) > 0:
        #         print 'Found plan!'
        #         break
        #     else:
        #         print 'Unable to find plan. Retrying...'
        # import pdb; pdb.set_trace()
        # visualize_path(dummy_plan, dummy_translation, planned_path_pub, goal_pos_pub)

        # # import pdb; pdb.set_trace()

        # raw_input('press enter to go to dummy position \n')
        # planner.execute_plan(dummy_plan)

        easy_rotation = RigidTransform.rotation_from_quaternion(np.array([0.012, -0.172, 0.984, -0.025]))

        # intermediate position ######################################################
        # import pdb; pdb.set_trace()
        final_rotation = T_grasp_world.rotation
        final_translation = T_grasp_world.translation
        z = T_grasp_world.z_axis
        intermediate_trans = final_translation - 0.001 * z
        intermediate_tf = RigidTransform(easy_rotation, intermediate_trans, from_frame='right_gripper', to_frame='base')

        # import pdb; pdb.set_trace()
        intermediate_pose_stamped = create_pose_stamped_from_rigid_tf(intermediate_tf, 'base')
        intermediate_plan = plan_to_pose_custom(intermediate_pose_stamped, planner)
        visualize_path(intermediate_plan, intermediate_trans, planned_path_pub, goal_pos_pub)

        # import pdb; pdb.set_trace()

        raw_input('press enter to go to intermediate position \n')
        planner.execute_plan(intermediate_plan)

        # final position ######################################################
        
        easy_T_grasp_world = RigidTransform(easy_rotation, T_grasp_world.translation, from_frame='right_gripper', to_frame='base')

        final_pose_stamped = create_pose_stamped_from_rigid_tf(easy_T_grasp_world, 'base')
        final_plan = plan_to_pose_custom(final_pose_stamped, planner)
        visualize_path(final_plan, T_grasp_world.translation, planned_path_pub, goal_pos_pub)
        
        # import pdb; pdb.set_trace()

        raw_input('press enter to go to final pose')
        planner.execute_plan(final_plan)

        # close gripper ######################################################

        raw_input('press enter to close the gripper')
        close_gripper()

        # lift ######################################################

        lift_trans = final_translation
        lift_trans[2] += 0.1
        lift_tf = RigidTransform(final_rotation, lift_trans, from_frame='right_gripper', to_frame='base')

        lift_pose_stamped = create_pose_stamped_from_rigid_tf(lift_tf, 'base')
        lift_plan = plan_to_pose_custom(lift_pose_stamped, planner)
        visualize_path(lift_plan, lift_trans, planned_path_pub, goal_pos_pub)
        
        # import pdb; pdb.set_trace()

        raw_input('press enter to lift the object')
        planner.execute_plan(lift_plan)    

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

    if ros_enabled:
        rospy.init_node('lab2_node')

        planned_path_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        goal_pos_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)


    # Mesh loading and pre-processing
    print 'Loading mesh'
    mesh = trimesh.load_mesh('objects/{}.obj'.format(args.obj))
    print 'Looking up transformation to the object.'
    T_obj_world = lookup_transform(args.obj)
    T_world_ar = lookup_transform('ar_marker_1') # TODO
    # mesh.apply_transform(T_obj_world.matrix)
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
        T_world_ar,
        args.obj
    )

    print 'Computing grasps.'

    # Each grasp is represented by T_grasp_world, a RigidTransform defining the 
    # position of the end effector
    T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)

    # Execute each grasp on the baxter / sawyer
    if args.baxter:
        gripper = baxter_gripper.Gripper(args.arm)
        planner = PathPlanner('{}_arm'.format(args.arm))

        for T_grasp_world in T_grasp_worlds:
            repeat = True
            while repeat:
                execute_grasp(T_grasp_world, planner, gripper, planned_path_pub, goal_pos_pub)
                repeat = raw_input("repeat? [y|n] ") == 'y'
