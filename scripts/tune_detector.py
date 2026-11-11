import cv2

from ball_tracking_machine_vision.ball_detector import detect_ball
from ball_tracking_machine_vision.camera_capture import open_camera


"""Live view with sliders for tuning HSV thresholds and Hough parameters."""


def main():
    # Select camera 0 is webcam, right now 4 is zed camera
    source = "/dev/video0" # Example: "/dev/video4"
    
    cap = open_camera(source, resolution=(1280, 720), fps=30)
    

    # Error if video source cant be opened
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # Method to test either color or hough
    method = "color"

    # Code to read frames continously
    try:
        while True:
            # Read next frame
            ok, frame = cap.read()
            if not ok:
                break

            # Returns (u, v, radius) in pixels, or None.
            result = detect_ball(frame, method=method)

            # Draw on a copy so the original frame stays unchanged.
            display = frame.copy()

            if result is not None:
                # u increases rightward; v increases downward.
                u, v, radius = result

                # Convert to integer for pixel coordinates
                center = (round(u), round(v))

                # Draw a green outline around the ball (B,G,R)
                cv2.circle(
                    display, center, round(radius), (0, 255, 0), 2
                )

                # Draw a red center dot; -1 means a filled circle.
                cv2.circle(display, center, 3, (0, 0, 255), -1)
                status = f"{method}: ball detected"
            else:
                status = f"{method}: no ball detected"


            # Display the method and detection status near the top left.
            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2,
            )
            cv2.imshow("Live ball detection", display)



            # q: quit, c: color detection, h: Hough detection
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                method = "color"
            elif key == ord("h"):
                method = "hough"
    finally:
        # Free camera and close 
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()