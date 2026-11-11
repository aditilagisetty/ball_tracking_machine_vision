"""Turning a pair of 2D detections into a 3D ball position."""
import cv2
import numpy as np

def compute_disparity(left_point, right_point):
    """Calculate the horizontal pixel difference between stereo detections.
    Args:
        left_point: Left-image detection (u, v, radius)
        right_point: Right-image detection (u, v, radius)

    Returns:
        Left u minus right u, in pixels
    """
    # ball detection returns u,v,radius. We want to difference between u values
    # in left and right image
    if left_point is None or right_point is None:
        return None
    distance = left_point[0]-right_point[0]
    return distance

def triangulate(left_point, right_point, P1,P2):
    """Reconstruct the ball's 3D position from rectified stereo detections.

    Uses the detected centers and stereo projection matrices to calculate
    a homogeneous 3D point, then divides by its scale factor W to obtain
    Cartesian coordinates.

    Args:
        left_point: Rectified left-image detection (u, v, radius)
        right_point: Rectified right-image detection (u, v, radius)
        P1: Left camera's 3x4 projection matrix from stereo rectification.
        P2: Right camera's 3x4 projection matrix from stereo rectification.

    Returns:
        An (x, y, z) tuple in the rectified left-camera frame, where x
        points right, y points down, and z points forward. Units are meters
        when the calibration translation is in meters.
    """
    if left_point is None or right_point is None:
        return None

    # Get center coordinates 
    left_uv = np.array(left_point[:2], dtype=np.float64).reshape(2,1)
    right_uv = np.array(right_point[:2], dtype=np.float64).reshape(2,1)

    # Check for finite coordinates
    if not (np.all(np.isfinite(left_uv))
            and np.all(np.isfinite(right_uv))):
        return None

    # Triangulate and give reconstructed 3D points in homogenous coordiantes
        # cv2.triangulatePoints returns X,Y,Z, and W, the scale factor in homogenous coordinates 
    points_4d = cv2.triangulatePoints(P1,P2,left_uv,right_uv)
    W = points_4d[3,0]

    # Check for finite points; make sure W is not near 0
    if not np.all(np.isfinite(points_4d)) or abs(W) < 1e-12:
        return None
    
    # Get points in 3D by dividing x,y,z coordinates by W
    points_3d = points_4d[:3,0] / W

    # Check for finite points
    if not np.all(np.isfinite(points_3d)):
        return None
    # Reject reconstructions on or behind either rectified camera plane.
    point_h = np.append(points_3d, 1.0)
    if (P1 @ point_h)[2] <= 0 or (P2 @ point_h)[2] <= 0:
        return None
    x,y,z = points_3d

    return x,y,z


def camera_to_world(point, camera_pose):
    """Transform a camera-frame point into the floor/world frame.
    
    Args:
        point: (x, y, z) in meters
        camera_pose: (R_camera_to_world, camera_position_world).
            R_camera_to_world: 3x3 rotation matrix mapping rectified
                camera axes into world axes.
            camera_position_world: Camera origin's (x, y, z)
                position in the world, in meters.
    
    Returns:
        World-frame x, y, and z 
    """

    # world position = camera-to-world rotation × camera position + camera’s world position    
    if point is None: 
        return None
    # Camera pose gives amount to rotate and translate
    rotation, translation = camera_pose

    # Convert inputs to arrays and reshape
    rotation = np.asarray(rotation).reshape(3,3)
    translation = np.asarray(translation).reshape(3)
    camera_point = np.asarray(point).reshape(3)


    # Rotate into world-axis directions, then shift to the world origin.
        # @ indicates matrix multiplication
    world_point = (rotation @ camera_point) + translation
    # Return world x,y,z coordinates

    world_x, world_y, world_z = world_point
    return world_x, world_y, world_z