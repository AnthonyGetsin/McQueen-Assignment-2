"""
Phase 4: EAR Calculation

This script implements:
- Implement Euclidean distance function
- Calculate EAR for both eyes
- Compute average EAR
- Display EAR value on screen
- Deliverable: Real-time EAR computation
"""

import cv2
import sys
import numpy as np

# Try to import MediaPipe (may not be available for Python 3.13+)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("⚠ Warning: MediaPipe is not available for this Python version")
    print("  MediaPipe may not support Python 3.13+ yet")
    print("  Consider using Python 3.11 or 3.12 for full MediaPipe support")


class EARCalculator:
    """Calculate Eye Aspect Ratio (EAR) from eye landmarks."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh and EAR calculator."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required but not available. "
                            "Please install it or use Python 3.11/3.12.")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        print("✓ EAR Calculator initialized")
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: Tuple (x, y) or numpy array
            point2: Tuple (x, y) or numpy array
        
        Returns:
            float: Euclidean distance
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.linalg.norm(p1 - p2)
    
    def calculate_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio (EAR).
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Where:
        - p1, p4 = horizontal eye corners (left, right)
        - p2, p3, p5, p6 = vertical eye landmarks (top and bottom)
        - || || represents Euclidean distance
        
        Args:
            eye_landmarks: List of 6 (x, y) coordinates for eye landmarks
                          [p1, p2, p3, p4, p5, p6]
        
        Returns:
            float: EAR value (0.0 if invalid input)
        """
        if len(eye_landmarks) != 6:
            return 0.0
        
        # Convert to numpy array for easier calculation
        points = np.array(eye_landmarks)
        
        # Calculate vertical distances
        # p2 to p6 (top to bottom, first vertical pair)
        vertical_1 = self.euclidean_distance(points[1], points[5])
        
        # p3 to p5 (top to bottom, second vertical pair)
        vertical_2 = self.euclidean_distance(points[2], points[4])
        
        # Calculate horizontal distance
        # p1 to p4 (left to right corner)
        horizontal = self.euclidean_distance(points[0], points[3])
        
        # Avoid division by zero
        if horizontal == 0:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """
        Extract specific eye landmarks and convert to pixel coordinates.
        
        Args:
            landmarks: MediaPipe landmark list
            indices: List of landmark indices to extract
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
        
        Returns:
            List of (x, y) pixel coordinates
        """
        eye_points = []
        for idx in indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            eye_points.append((x, y))
        return eye_points
    
    def process_frame(self, frame):
        """
        Process a single frame and calculate EAR.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            annotated_frame: Frame with EAR values displayed
            left_ear: Left eye EAR value or None
            right_ear: Right eye EAR value or None
            avg_ear: Average EAR value or None
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        left_ear = None
        right_ear = None
        avg_ear = None
        
        if results.multi_face_landmarks:
            # Face detected - calculate EAR
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye = self.get_eye_landmarks(
                    face_landmarks.landmark,
                    self.LEFT_EYE_INDICES,
                    w, h
                )
                right_eye = self.get_eye_landmarks(
                    face_landmarks.landmark,
                    self.RIGHT_EYE_INDICES,
                    w, h
                )
                
                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Draw eye landmarks for visualization
                for point in left_eye + right_eye:
                    cv2.circle(annotated_frame, point, 2, (0, 255, 0), -1)
        
        return annotated_frame, left_ear, right_ear, avg_ear


def run_ear_calculation():
    """Run real-time EAR calculation."""
    print("\n" + "=" * 50)
    print("Phase 4: EAR Calculation")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Make sure your face is visible to the camera")
    print("\nEAR Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)")
    print("Typical EAR values: 0.15-0.35 (higher = more open)")
    
    # Initialize calculator
    try:
        calculator = EARCalculator()
    except ImportError as e:
        print(f"✗ {e}")
        return False
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return False
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam: {width}x{height} @ {fps} FPS")
    print("Starting EAR calculation...")
    
    frame_count = 0
    ear_calculated_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for EAR calculation
            annotated_frame, left_ear, right_ear, avg_ear = calculator.process_frame(frame)
            
            # Display EAR values
            y_offset = 30
            if left_ear is not None and right_ear is not None and avg_ear is not None:
                ear_calculated_count += 1
                
                # Display EAR values with color coding
                cv2.putText(annotated_frame, f"Left EAR: {left_ear:.3f}", 
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Right EAR: {right_ear:.3f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Average EAR: {avg_ear:.3f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 40
                
                # Display status
                status = "Face Detected"
                status_color = (0, 255, 0)  # Green
            else:
                status = "No Face Detected"
                status_color = (0, 0, 255)  # Red
            
            cv2.putText(annotated_frame, status, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += 35
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(annotated_frame, "Press 'q' to quit",
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Phase 4: EAR Calculation', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ EAR calculated in {ear_calculated_count} frames")
        if frame_count > 0:
            print(f"✓ Calculation rate: {100 * ear_calculated_count / frame_count:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    try:
        if not MEDIAPIPE_AVAILABLE:
            print("✗ MediaPipe is required for Phase 4")
            print("  Please install MediaPipe or use Python 3.11/3.12")
            sys.exit(1)
        
        success = run_ear_calculation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

