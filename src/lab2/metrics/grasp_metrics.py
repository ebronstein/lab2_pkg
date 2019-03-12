#!/home/cc/ee106b/sp19/class/ee106b-aap/ee106b_sp19/ros_workspaces/lab2_ws/env/bin/python -W ignore::DeprecationWarning
"""
Grasp Metrics for EE106B grasp planning lab
Author: Chris Correa
"""
# may need more imports
import numpy as np
from lab2.utils import vec, adj, look_at_general, make_homo, wrench_basis
import cvxpy as cp
import math

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Compute the force closure of some object at contacts, with normal vectors 
    stored in normals You can use the line method described in HW2.  if you do you 
    will not need num_facets

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE
    c1, c2 = vertices
    n1, n2 = normals
    v = c2 - c1
    theta = np.arctan(mu)
    
    # binary metric: 1 = in friction cone, 0 = not in friction cone
    # cos_theta = np.cos(theta)
    # in_fc1 = np.inner(-v, n1) / np.linalg.norm(v) <= cos_theta
    # in_fc2 = np.inner(v, n2) / np.linalg.norm(v) <= cos_theta
    # return float(in_fc1 and in_fc2)

    cos_angle1 = np.inner(-v, n1) / np.linalg.norm(v)
    cos_angle2 = np.inner(v, n2) / np.linalg.norm(v)
    # best cos_angle: 1; worst cos_angle: -1
    # sum of cos angles in [-2, 2]
    return (2 + cos_angle1 + cos_angle2) / 4.

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """ 
    defined in the book on page 219.  Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient

    Returns
    -------
    :obj:`numpy.ndarray` grasp map
    """
    # YOUR CODE HERE
    v1, v2 = vertices
    n1, n2 = normals
    g1 = look_at_general(v1, n1)
    g2 = look_at_general(v2, n2)
    adj_g1 = adj(g1)
    adj_g2 = adj(g2)
    G1 = adj_g1.dot(wrench_basis) # 6x4
    G2 = adj_g2.dot(wrench_basis) # 6x4
    return np.hstack([G1, G2]) # 6x8


def contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute whether the given grasp (at contacts with surface normals) can produce 
    the desired_wrench.  will be used for gravity resistance. 

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : :obj:`numpy.ndarray`
        potential wrench to be produced

    Returns
    -------
    bool : whether contact forces can produce the desired_wrench on the object
    """
    # YOUR CODE HERE
    
    grasp_map = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    m, n = grasp_map.shape # 6x8

    f1 = cp.Variable()
    f2 = cp.Variable()
    f3 = cp.Variable()
    f4 = cp.Variable()
    f5 = cp.Variable()
    f6 = cp.Variable()
    f7 = cp.Variable()
    f8 = cp.Variable()

    f = cp.Variable(n) # 8
    objective = cp.Minimize(cp.norm(
            grasp_map[:, 0] * f1 +
            grasp_map[:, 1] * f2 +
            grasp_map[:, 2] * f3 +
            grasp_map[:, 3] * f4 +
            grasp_map[:, 4] * f5 +
            grasp_map[:, 5] * f6 +
            grasp_map[:, 6] * f7 +
            grasp_map[:, 7] * f8 - 
            desired_wrench))
    
    constraints = [
                    f3 >= 0,
                    f7 >= 0,
                    f1**2 + f2**2 <= mu**2 * f3**2,
                    f5**2 + f6**2 <= mu**2 * f7**2,
                    cp.abs(f4) <= gamma * f3,
                    cp.abs(f8) <= gamma * f7
                  ]

    # constraints = [
    #                 f[2] >= 0, # force 1 z non-negative
    #                 f[6] >= 0, # force 2 z non-negative
    #                 # f[0]**2 + f[1]**2 <= mu**2 * f[2]**2, # force 1 in FC

    #                 cp.sqrt(f[0]**2 + f[1]**2) <= mu * f[2], # force 1 in FC

    #                 cp.abs(f[3]) <= gamma * f[2], # force 1 
    #                 # f[4]**2 + f[5]**2 <= mu**2 * f[6]**2, # force 2 in FC
    #                 # cp.abs(f[7]) <= gamma * f[6] # force 2
    #               ]
    prob = cp.Problem(objective, constraints)
    # import pdb; pdb.set_trace()
    prob.solve()
    return prob.status == cp.OPTIMAL


def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Gravity produces some wrench on your object.  Computes whether the grasp can 
    produce and equal and opposite wrench

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will 
        be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE (contact forces exist may be useful here)
    desired_wrench = np.array([0, 0, -9.8 * object_mass, 0, 0, 0])
    return contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench)

def compute_custom_metric(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    I suggest Ferrari Canny, but feel free to do anything other metric you find. 

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will 
        be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE :)
    raise NotImplementedError