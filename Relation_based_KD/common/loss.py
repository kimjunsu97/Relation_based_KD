
import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def distance_KD(predicted, target):
    """
    Distance error for Knowledge Distillation
    """
    assert predicted.shape == target.shape

    predicted = torch.squeeze(predicted)
    target = torch.squeeze(target)
    joint_list = [[0,1],[0,4],[0,7],[1,2],[2,3],[4,5],[5,6],[7,8],[8,9],[8,11],[8,14],[9,10],[11,12],[12,13],[14,15],[15,16]]
    distance_pred_list = []
    distance_target_list = []
    for joint_indices in joint_list:
        predicted_joints = predicted[:,joint_indices,:]
        pred_v = predicted_joints[:,1] - predicted_joints[:, 0]
        pred_norm_v = torch.norm(pred_v, dim=1)  # v1 벡터들의 크기 계산
        pred_norm_v = torch.unsqueeze(pred_norm_v,dim=1)
        distance_pred_list.append(pred_norm_v)
                
        target_joints = target[:,joint_indices,:]
        target_v = target_joints[:,1] - target_joints[:, 0]
        
        target_norm_v = torch.norm(target_v, dim=1)  # v1 벡터들의 크기 계산
        target_norm_v = torch.unsqueeze(target_norm_v,dim=1)
        distance_target_list.append(target_norm_v)

    pred_stacked_distance = torch.cat(distance_pred_list,dim=1)
    target_stacked_distance = torch.cat(distance_target_list,dim=1)

    return torch.mean(torch.norm(pred_stacked_distance - target_stacked_distance, dim=-1))

def angle_KD(predicted, target):
    """
    angle error for Knowledge Distillation
    """
    assert predicted.shape == target.shape

    predicted = torch.squeeze(predicted)
    target = torch.squeeze(target)
    #joint_list = [[0,1,2],[0,4,5],[0,7,8],[1,2,3],[4,5,6],[8,9,10],[8,11,12],[8,14,15],[11,12,13],[14,15,16]]
    joint_list = [[0,1,2],[0,4,5],[0,7,8],[1,2,3],[4,5,6],[7,8,9],[8,11,12],[8,14,15],[11,12,13],[14,15,16]]
    angle_pred_list = []
    angle_target_list = []
    
    for joint_indices in joint_list:
        predicted_joints = predicted[:,joint_indices,:]
        
        pred_v1 = predicted_joints[:,1] - predicted_joints[:, 0]
        pred_v2 = predicted_joints[:,2] - predicted_joints[:,0]
        
        pred_cross_product = torch.cross(pred_v1, pred_v2)
        pred_dot_product = torch.einsum('ij,ij->i', pred_v1, pred_v2)  # dot product computation
        
        pred_angles = torch.atan2(torch.norm(pred_cross_product, dim=1), pred_dot_product)
        pred_angles = torch.unsqueeze(pred_angles, dim=1)
        angle_pred_list.append(pred_angles)

        target_joints = target[:,joint_indices,:]
        
        target_v1 = target_joints[:,1] - target_joints[:, 0]
        target_v2 = target_joints[:,2] - target_joints[:,0]
        
        target_cross_product = torch.cross(target_v1, target_v2)
        target_dot_product = torch.einsum('ij,ij->i', target_v1, target_v2)  # dot product computation

        target_angles = torch.atan2(torch.norm(target_cross_product, dim=1), target_dot_product)
        target_angles = torch.unsqueeze(target_angles,dim=1)
        angle_target_list.append(target_angles)

    pred_stacked_angle = torch.cat(angle_pred_list,dim=1)
    target_stacked_angle = torch.cat(angle_target_list,dim=1)

    return torch.mean(torch.norm(pred_stacked_angle - target_stacked_angle, dim=-1))
