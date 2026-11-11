"""Interactive floor mapping from a rectified left-camera checkerboard image."""
import json
from pathlib import Path
import cv2
import numpy as np


def solve_floor_pose(corners, pattern, square_size, selected, K):
    """Use three selected grid extremes to remove square-board orientation ambiguity."""
    columns, rows = pattern
    grid = np.array([(c, r) for r in range(rows) for c in range(columns)], dtype=float)
    origin, along_x, along_y = grid[selected]
    extremes = {(0, 0), (columns - 1, 0), (0, rows - 1), (columns - 1, rows - 1)}
    if len(set(selected)) != 3 or any(tuple(p) not in extremes for p in (origin, along_x, along_y)):
        raise ValueError('Select three different extreme INTERNAL corners.')
    dx, dy = along_x - origin, along_y - origin
    if abs(dx @ dy) > 1e-6:
        raise ValueError('Second and third corners must be adjacent to the first, not diagonal.')
    # Internal corners begin one square away from the outer board origin.
    objects = np.zeros((len(grid), 3), dtype=np.float64)
    objects[:, 0] = ((grid - origin) @ (dx / np.linalg.norm(dx)) + 1) * square_size
    objects[:, 1] = ((grid - origin) @ (dy / np.linalg.norm(dy)) + 1) * square_size
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    # Images are already rectified, so use P1's intrinsic block and ZERO distortion.
    ok, rvec, tvec = cv2.solvePnP(objects, image_points, K, np.zeros(5))
    if not ok:
        raise ValueError('Could not estimate board pose.')
    R, _ = cv2.Rodrigues(rvec)
    camera_position = (-R.T @ tvec).reshape(3)
    if camera_position[2] <= 0 or np.any((R @ objects.T + tvec)[2] <= 0):
        raise ValueError('Pose has an invalid orientation; check the selected corners.')
    projected, _ = cv2.projectPoints(objects, rvec, tvec, K, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points)**2, axis=1))))
    if not np.isfinite(rms) or rms > 3:
        raise ValueError(f'Corner reprojection error is too large: {rms:.2f} pixels.')
    return (R.T, camera_position), rvec, tvec, rms


def capture_world_pose(image, P1, pattern, square_size, output_path):
    """Freeze a frame, identify axes, review, then save a camera-to-world pose."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, pattern)
    if not found:
        print('Full checkerboard not found. Improve lighting/visibility and press m again.')
        return None
    window = 'Map floor - click INTERNAL corners'
    selected = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 3:
            distance = np.linalg.norm(corners[:, 0] - [x, y], axis=1)
            index = int(np.argmin(distance))
            if distance[index] <= 25 and index not in selected:
                selected.append(index)
    # Autosize keeps mouse coordinates aligned with the actual image pixels.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    prompts = ['Click nearest-left INTERNAL corner',
               'Click nearest-right INTERNAL corner (+x)',
               'Click far-left INTERNAL corner (+y)']
    candidate = None
    error = None
    try:
        while True:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern, corners, True)
            for i, index in enumerate(selected):
                center = tuple(np.rint(corners[index, 0]).astype(int))
                cv2.circle(preview, center, 7, (0, 255, 255), 2)
                cv2.putText(preview, str(i + 1), center, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            if len(selected) == 3 and candidate is None and error is None:
                try:
                    candidate = solve_floor_pose(corners, pattern, square_size, selected, P1[:, :3])
                    pose, rvec, tvec, rms = candidate
                    print(f'Camera world position (m): {pose[1]}; reprojection RMS: {rms:.3f}px')
                    print('Sanity check: expected height about 0.75m and horizontal distance about 1m.')
                    print('Review axes: red +x right, green +y into board, blue +z up. Enter saves; u retries; Esc cancels.')
                except ValueError as exc:
                    error = str(exc)
                    print(error)
            if candidate:
                _, rvec, tvec, rms = candidate
                cv2.drawFrameAxes(preview, P1[:, :3], np.zeros(5), rvec, tvec, square_size * 3)
                message = f'Enter: save | u: retry | Esc: cancel | RMS {rms:.2f}px'
            elif error:
                message = 'Invalid selection: u to retry; Esc cancels'
            else:
                message = prompts[len(selected)] + ' | Esc cancels'
            cv2.putText(preview, message, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord('q')):
                return None
            if key == ord('u'):
                selected.clear(); candidate = None; error = None
            if key in (10, 13) and candidate:
                pose = candidate[0]
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Enter explicitly accepts replacing this pose file.
                data = {'source_frame': 'rectified_left_camera',
                        'image_size': [image.shape[1], image.shape[0]],
                        'R_camera_to_world': pose[0].tolist(),
                        'camera_position_world': pose[1].tolist(),
                        'square_size_m': square_size, 'pattern_size': list(pattern),
                        'world_axes': 'x right along board, y into board, z up',
                        'origin': 'nearest-left outer board corner',
                        'reprojection_rms_px': candidate[3]}
                path.write_text(json.dumps(data, indent=2) + '\n')
                print(f'Saved world pose: {path.resolve()}')
                return pose
    finally:
        cv2.destroyWindow(window)
