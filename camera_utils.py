import cv2
import numpy as np
import mediapipe as mp

class CameraCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
    def capture_image(self):
        """Capture an image from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from camera")
        return frame
    
    def get_person_keypoints(self, frame):
        """Get person keypoints using MediaPipe Pose."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        return results.pose_landmarks
        
    def calculate_distance_to_camera(self, frame, landmarks):
        """Calculate approximate distance to camera based on person height in pixels."""
        if landmarks is None:
            return None
            
        # Get image dimensions
        image_height, image_width, _ = frame.shape
        
        # Get nose and ankle points
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate height in pixels
        ankle_y = min(left_ankle.y, right_ankle.y)
        height_pixels = (ankle_y - nose.y) * image_height
        
        # Approximate distance using the known height (assumed 170cm average height)
        # focal_length * real_height = pixel_height * distance
        focal_length = 1000  # approximate focal length for webcam
        real_height = 170  # average height in cm
        distance = (focal_length * real_height) / height_pixels
        
        return distance
        
    def release(self):
        """Release the camera."""
        self.cap.release()
        cv2.destroyAllWindows()
