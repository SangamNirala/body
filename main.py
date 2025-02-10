import cv2
import numpy as np
from camera_utils import CameraCapture
from body_measurements import BodyMeasurements

def main():
    # Initialize camera and measurement systems
    camera = CameraCapture()
    measurements = BodyMeasurements()
    
    print("Stand 6-7 feet away from the camera.")
    print("Press 'c' to capture or 'q' to quit")
    
    while True:
        # Capture frame
        frame = camera.capture_image()
        
        # Get keypoints
        landmarks = camera.get_person_keypoints(frame)
        
        # Draw guide overlay
        height, width = frame.shape[:2]
        cv2.line(frame, (width//3, 0), (width//3, height), (0, 255, 0), 1)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (0, 255, 0), 1)
        
        # Display instructions
        cv2.putText(frame, "Stand between the green lines", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Body Measurement', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Calculate distance
            distance = camera.calculate_distance_to_camera(frame, landmarks)
            
            if distance is None:
                print("Could not detect person. Please try again.")
                continue
                
            # Calculate measurements
            height = measurements.calculate_height(frame, distance)
            waist = measurements.calculate_waist(frame)
            
            if height is not None and waist is not None:
                print("\nMeasurements:")
                print("Height: {:.1f} cm".format(height))
                print("Waist: {:.1f} cm".format(waist))
            else:
                print("Could not calculate measurements. Please try again.")
    
    # Cleanup
    camera.release()

if __name__ == "__main__":
    main()
