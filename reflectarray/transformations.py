import numpy as np
from scipy import special as sp

def spherical_to_cartesian_vector(vec, theta_list, phi_list):
    # vec n X 3, theta_list n X 1, phi_list n X 1
    if np.isscalar(theta_list):
        n = 1
    else:
        n = phi_list.size
    T = np.reshape(np.column_stack((
        np.sin(theta_list)*np.cos(phi_list), 
        np.cos(theta_list)*np.cos(phi_list),
        -np.sin(phi_list),
        np.sin(theta_list)*np.sin(phi_list),
        np.cos(theta_list)*np.sin(phi_list),
        np.cos(phi_list),
        np.cos(theta_list),
        -np.sin(theta_list),
        np.zeros(n))),
              (n, 3, 3))
    return np.matmul(T, vec[:,:,None])[:,:,0]

def cartesian_to_spherical_vector(vec, theta_list, phi_list):
    # vec n X 3, theta_list n X 1, phi_list n X 1
    if np.isscalar(theta_list):
        n = 1
    else:
        n = phi_list.size
    T = np.reshape(np.column_stack((
        np.sin(theta_list)*np.cos(phi_list), 
        np.sin(theta_list)*np.sin(phi_list),
        np.cos(theta_list),
        np.cos(theta_list)*np.cos(phi_list),
        np.cos(theta_list)*np.sin(phi_list),
        -np.sin(theta_list),
        -np.sin(phi_list),
        np.cos(phi_list),
        np.zeros(n))),
              (n, 3, 3))
    return np.matmul(T, vec[:,:,None])[:,:,0]

def cylindrical_to_cartesian_vector(vec, phi_list):
    # vec n X 3, phi_list n X 1
    if np.isscalar(phi_list):
        n = 1
    else:
        n = phi_list.size
    T = np.reshape(np.column_stack((
        np.cos(phi_list),
        -np.sin(phi_list),
        np.zeros(n),
        np.sin(phi_list),
        np.cos(phi_list),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n))),
              (n, 3, 3))
    out_temp = np.matmul(T, vec[:,:,None])[:,:,0]
    return np.transpose(np.array([out_temp[:,0], out_temp[:,1], out_temp[:,2]]))

def cartesian_to_cylindrical_vector(vec, phi_list):
    # vec n X 3, phi_list n X 1
    if np.isscalar(phi_list):
        n = 1
    else:
        n = phi_list.size
    
    vec_reshape = np.transpose(np.array([vec[:,0], vec[:,1], vec[:,2]]))
    T = np.reshape(np.column_stack((
        np.cos(phi_list),
        np.sin(phi_list),
        np.zeros(n),
        -np.sin(phi_list),
        np.cos(phi_list),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n))),
              (n, 3, 3))
    return np.matmul(T, vec_reshape[:,:,None])[:,:,0]

def cartesian_to_circular_vector(vec):
    # vec n X 3
    vec_reshape = np.transpose(np.array([vec[:,0], vec[:,2]]))
    T_circular = np.array([[1/np.sqrt(2), -1j/np.sqrt(2)], [1/np.sqrt(2), 1j/np.sqrt(2)]])
    return np.matmul(T_circular[None,:,:], vec_reshape[:,:,None])[:,:,0]

def spherical_to_circular_vector(vec):
    # vec n X 3
    vec_reshape = np.transpose(np.array([vec[:,1], vec[:,2]]))
    T_circular = np.array([[1/np.sqrt(2), -1j/np.sqrt(2)], [1/np.sqrt(2), 1j/np.sqrt(2)]])
    return np.matmul(T_circular[None,:,:], vec_reshape[:,:,None])[:,:,0]

def rotate_vector(vector_array, angle, rotation_axis):
    if vector_array.ndim == 1:
        vector_array = vector_array[None,:]
        
    angle = np.radians(angle)[:,None]
    
    axis_dictionary = {'x': np.array([1, 0, 0]), 'y': np.array([0, 1, 0]), 'z': np.array([0, 0, 1])}
    
    if isinstance(rotation_axis, str):
        if rotation_axis in ['x', 'y', 'z']:
            rotation_axis = axis_dictionary[rotation_axis][None,:]
        
        else:
            raise Exception('Specify rotation axis as \'x\', \'y\', or \'z\', or supply arbitrary length (3,) vector.')
            
    else:
        if rotation_axis.ndim == 1:
            rotation_axis = rotation_axis[None,:]
        
    out = (np.sum(rotation_axis * vector_array, axis=1, keepdims=True) * rotation_axis +
            np.cross(np.cross(rotation_axis, vector_array, axis=1), rotation_axis, axis=1) * np.cos(angle) +
            np.cross(rotation_axis, vector_array, axis=1) * np.sin(angle))
    return out
    
