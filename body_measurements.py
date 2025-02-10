import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class BodyMeasurements:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        # Load YOLOv8 model for person detection
        self.yolo_model = YOLO('yolov8n.pt')
        
    def calculate_height(self, frame, distance_to_camera):
        """Calculate person's height using pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Get image dimensions
        image_height, image_width, _ = frame.shape
        
        # Get nose and ankle points
        nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate height in pixels
        ankle_y = min(left_ankle.y, right_ankle.y)
        height_pixels = (ankle_y - nose.y) * image_height
        
        # Convert pixels to cm using the distance
        focal_length = 1000  # approximate focal length for webcam
        height_cm = (height_pixels * distance_to_camera) / focal_length
        
        return height_cm
        
    def calculate_waist(self, frame):
        """Calculate person's waist measurement using pose landmarks and contour analysis."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Get relevant landmarks for waist measurement
        left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate waist region
        image_height, image_width, _ = frame.shape
        waist_y = int((left_hip.y + right_hip.y) * image_height / 2)
        
        # Create a mask for the person's body
        results = self.yolo_model(frame)
        person_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_mask[y1:y2, x1:x2] = 255
        
        # Get waist contour
        waist_line = person_mask[waist_y, :]
        waist_points = np.where(waist_line > 0)[0]
        
        if len(waist_points) < 2:
            return None
            
        # Calculate waist width in pixels
        waist_width_pixels = waist_points[-1] - waist_points[0]
        
        # Convert to approximate cm (assuming average waist depth)
        pixel_to_cm_ratio = 0.5  # This would need proper calibration
        waist_circumference = waist_width_pixels * pixel_to_cm_ratio * np.pi
        
        return waist_circumference
