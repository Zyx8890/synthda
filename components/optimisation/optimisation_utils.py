import numpy as np
from scipy.interpolate import interp1d

# converts npz to npy
def map_h36m_to_smpl(real_path_npz):

    real_data = np.load(real_path_npz)
    
    # restructuring the array
    h36m_joints = real_data['reconstruction']
    
    """
    Maps Human3.6M 17 keypoints to SMPL 24 keypoints.
    :param h36m_joints: (N, 17, 3) NumPy array of 3D keypoints
    :return: (N, 24, 3) NumPy array of mapped keypoints
    """
    smpl_joints = np.zeros((h36m_joints.shape[0], 22, 3))

    # Mapping H36M to SMPL keypoints
    h36m_to_smpl = {
        0: 0,  # Pelvis 
        1: 4,  # L_Hip (H36M Left Hip)
        2: 1,  # R_Hip (H36M Right Hip)
        3: 7,  # Spine1 (H36M Spine)
        4: 5,  # L_Knee (H36M Left Knee)
        5: 2,  # R_Knee (H36M Right Knee)
        6: 8,  # Spine2 (H36M Neck→calculate)
        7: 6,  # L_Ankle (H36M Left Ankle)
        8: 3,  # R_Ankle (H36M Right Ankle)
        9: 8,  # Spine3 (calculate)
        12: 8, # Neck (H36M Neck)
        13: 11, # L_Collar (H36M Left Shoulder)
        14: 14, # R_Collar (H36M Right Shoulder)
        15: 10, # Head (H36M Head)
        16: 11, # L_Shoulder (H36M Left Shoulder)
        17: 14, # R_Shoulder (H36M Right Shoulder)
        18: 12, # L_Elbow (H36M Left Elbow)
        19: 15, # R_Elbow (H36M Right Elbow)
        20: 13, # L_Wrist (H36M Left Wrist)
        21: 16, # R_Wrist (H36M Right Wrist)
    }

    # Apply direct mappings
    for smpl_idx, h36m_idx in h36m_to_smpl.items():
        smpl_joints[:, smpl_idx, :] = h36m_joints[:, h36m_idx, :]

    # Compute Spine2 (midpoint of Spine1 and Spine3)
    smpl_joints[:, 6, :] = (smpl_joints[:, 3, :] + smpl_joints[:, 9, :]) / 2

    # Compute Spine3 (midpoint of Spine2 and Neck)
    smpl_joints[:, 9, :] = (smpl_joints[:, 6, :] + smpl_joints[:, 12, :]) / 2

    # L_Foot (same as L_Ankle)
    smpl_joints[:, 10, :] = smpl_joints[:, 7, :]

    # R_Foot (same as R_Ankle)
    smpl_joints[:, 11, :] = smpl_joints[:, 8, :]

    return smpl_joints

def upsample_pose_data(pose_data, target_frames):
    """
    Upsample pose data using linear interpolation to match the target number of frames.
    
    Args:
        pose_data (np.ndarray): Pose data of shape (T, J, 3),
                                where T is the number of frames, J is the number of joints.
        target_frames (int): Desired number of frames after upsampling.
    
    Returns:
        np.ndarray: Upsampled pose data with shape (target_frames, J, 3).
    """
    pose_data = np.load(pose_data)

    T, J, D = pose_data.shape  # Extract shape: T = frames, J = joints, D = 3 (3D coordinates)
    new_times = np.linspace(0, T - 1, target_frames)  # New frame timeline
    old_times = np.arange(T)  # Original frame timeline
    new_pose_data = np.empty((target_frames, J, D))  # Placeholder for upsampled data

    # Interpolate each joint and coordinate independently
    for j in range(J):
        for d in range(D):
            f = interp1d(old_times, pose_data[:, j, d], kind='linear', fill_value="extrapolate")
            new_pose_data[:, j, d] = f(new_times)

    return new_pose_data
    
def center_and_rotate_smpl(npy_path, output_path):
    """
    Load an .npy file (frames, joints, 3D), center each frame's geometric median as origin,
    rotate by 180 degrees along the X-axis, and save back the transformed coordinates.
    """

    # Load the joint positions from the .npy file (Shape: [Frames, Joints, 3])
    joint_data = np.load(npy_path)

    # Ensure correct shape (Frames, 22 joints, 3D)
    if joint_data.shape[1:] != (22, 3):
        raise ValueError(f"Expected shape (Frames, 22, 3), but got {joint_data.shape}")

    # Process each frame and center it
    centered_rotated_data = np.zeros_like(joint_data)

    # Rotation matrix for 180-degree rotation around the X-axis
    rotation_matrix = np.array([
        [1,  0,  0],  # X-axis unchanged
        [0, -1,  0],  # Flip Y-axis
        [0,  0, -1]   # Flip Z-axis
    ])

    # Compute the median center of all joints
    median_center = np.median(joint_data[0], axis=0)

    for frame_idx in range(joint_data.shape[0]):  # Iterate over frames
        frame_joints = joint_data[frame_idx]  # Shape (22, 3)

        # Shift all joints so that the median is at (0,0,0)
        centered_frame = frame_joints - median_center

        # Apply rotation to each joint
        rotated_frame = np.dot(centered_frame, rotation_matrix.T)  # Matrix multiplication

        # Store back into the new array
        centered_rotated_data[frame_idx] = rotated_frame

    # Save the new numpy array with centered and rotated data
    np.save(output_path, centered_rotated_data)

    # print(f"Centered and rotated data saved to {output_path}")


def compute_P_opt(real_pose_path, synthetic_pose_path, alpha, w_A, w_B):
    """
    Compute the optimal pose P_opt for each joint given the real and synthetic poses.
    
    Args:
        P_r (np.ndarray): Real pose data of shape (J, 3).
        P_s (np.ndarray): Synthetic pose data of shape (J, 3).
        alpha (float): Scaling factor.
        w_A (float): Weight for the real pose.
        w_B (float): Weight for the synthetic pose.
        
    Returns:
        np.ndarray: The optimal pose P_opt of shape (J, 3).
    """

    # # Load the real and synthetic pose data from .npy files
    P_r = np.load(real_pose_path)  # Shape: (J, 3)
    P_s = np.load(synthetic_pose_path)  # Shape: (J, 3)
    
    # Compute the weighted difference vector for each joint
    D_vector = w_A * P_r - w_B * P_s  # Shape: (J, 3)
    
    # Compute the norm (scalar distance) for each joint
    d = np.linalg.norm(D_vector, axis=1, keepdims=True)  # Shape: (J, 1)
    
    # Avoid division by zero by setting zero norms to one (will be multiplied by 0 anyway)
    d_safe = np.where(d == 0, 1, d)
    
    # Compute the unit direction vector for each joint
    u = D_vector / d_safe  # Shape: (J, 3)

    # Compute the optimal pose for each joint
    P_opt = P_r + alpha * d * u
    
    return P_opt
