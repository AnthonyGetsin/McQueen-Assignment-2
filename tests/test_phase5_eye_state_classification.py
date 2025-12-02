"""
Phase 5: Eye State Classification

This script implements:
- Implement threshold-based classification
- Add state labels (OPEN/CLOSED)
- Color-code display (green=open, red=closed)
- Test with deliberate blinking
- Deliverable: Working eye state classifier
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


class EyeStateClassifier:
    """Classify eye state (OPEN/CLOSED) based on EAR threshold."""
    
    def __init__(self, ear_threshold=0.21):
        """
        Initialize MediaPipe Face Mesh and set EAR threshold.
        
        Args:
            ear_threshold: Threshold for classification
                          EAR < threshold → CLOSED
                          EAR >= threshold → OPEN
        """
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
        
        # EAR threshold for classification
        # According to project requirements: EAR < 0.21 → CLOSED, EAR >= 0.21 → OPEN
        self.EAR_THRESHOLD = ear_threshold
        
        print("✓ Eye State Classifier initialized")
        print(f"  - EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"  - Classification: EAR < {self.EAR_THRESHOLD} → CLOSED, EAR >= {self.EAR_THRESHOLD} → OPEN")
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.linalg.norm(p1 - p2)
    
    def calculate_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio (EAR).
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        if len(eye_landmarks) != 6:
            return 0.0
        
        points = np.array(eye_landmarks)
        vertical_1 = self.euclidean_distance(points[1], points[5])
        vertical_2 = self.euclidean_distance(points[2], points[4])
        horizontal = self.euclidean_distance(points[0], points[3])
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def classify_eye_state(self, ear):
        """
        Classify eye state based on EAR threshold.
        
        Args:
            ear: Eye Aspect Ratio value
        
        Returns:
            str: "OPEN" or "CLOSED"
        """
        if ear < self.EAR_THRESHOLD:
            return "CLOSED"
        else:
            return "OPEN"
    
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """Extract specific eye landmarks and convert to pixel coordinates."""
        eye_points = []
        for idx in indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            eye_points.append((x, y))
        return eye_points
    
    def process_frame(self, frame):
        """
        Process a single frame and classify eye state.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            annotated_frame: Frame with classification results
            left_state: Left eye state ("OPEN"/"CLOSED") or None
            right_state: Right eye state ("OPEN"/"CLOSED") or None
            avg_ear: Average EAR value or None
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        left_state = None
        right_state = None
        avg_ear = None
        
        if results.multi_face_landmarks:
            # Face detected - classify eye state
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
                
                # Classify eye states
                left_state = self.classify_eye_state(left_ear)
                right_state = self.classify_eye_state(right_ear)
                
                # Draw eye landmarks with color coding
                # Green for open, red for closed
                left_color = (0, 255, 0) if left_state == "OPEN" else (0, 0, 255)
                right_color = (0, 255, 0) if right_state == "OPEN" else (0, 0, 255)
                
                for point in left_eye:
                    cv2.circle(annotated_frame, point, 3, left_color, -1)
                for point in right_eye:
                    cv2.circle(annotated_frame, point, 3, right_color, -1)
        
        return annotated_frame, left_state, right_state, avg_ear


def run_eye_state_classification():
    """Run real-time eye state classification."""
    print("\n" + "=" * 50)
    print("Phase 5: Eye State Classification")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Make sure your face is visible to the camera")
    print("\nTry blinking to see the classification change!")
    print("Green = OPEN, Red = CLOSED")
    
    # Initialize classifier
    try:
        classifier = EyeStateClassifier(ear_threshold=0.21)
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
    print("Starting eye state classification...")
    
    frame_count = 0
    classification_count = 0
    blink_count = 0
    previous_state = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for eye state classification
            annotated_frame, left_state, right_state, avg_ear = classifier.process_frame(frame)
            
            # Display classification results
            y_offset = 30
            
            if left_state is not None and right_state is not None and avg_ear is not None:
                classification_count += 1
                
                # Determine overall state (both eyes must be closed for "CLOSED")
                if left_state == "CLOSED" and right_state == "CLOSED":
                    overall_state = "CLOSED"
                    state_color = (0, 0, 255)  # Red
                    # Count blinks (transition from OPEN to CLOSED)
                    if previous_state == "OPEN":
                        blink_count += 1
                else:
                    overall_state = "OPEN"
                    state_color = (0, 255, 0)  # Green
                
                previous_state = overall_state
                
                # Display overall state (large, prominent)
                cv2.putText(annotated_frame, f"EYE STATE: {overall_state}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 3)
                y_offset += 50
                
                # Display individual eye states
                left_color = (0, 255, 0) if left_state == "OPEN" else (0, 0, 255)
                right_color = (0, 255, 0) if right_state == "OPEN" else (0, 0, 255)
                
                cv2.putText(annotated_frame, f"Left Eye: {left_state}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Right Eye: {right_state}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
                y_offset += 35
                
                # Display EAR value
                cv2.putText(annotated_frame, f"Average EAR: {avg_ear:.3f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
                
                # Display threshold
                cv2.putText(annotated_frame, f"Threshold: {classifier.EAR_THRESHOLD:.3f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 30
                
                # Display blink count
                cv2.putText(annotated_frame, f"Blinks: {blink_count}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 35
                
                status = "Face Detected"
                status_color = (0, 255, 0)  # Green
            else:
                status = "No Face Detected"
                status_color = (0, 0, 255)  # Red
                previous_state = None
            
            cv2.putText(annotated_frame, status, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y_offset += 30
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(annotated_frame, "Press 'q' to quit",
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Phase 5: Eye State Classification', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Classification performed in {classification_count} frames")
        print(f"✓ Total blinks detected: {blink_count}")
        if frame_count > 0:
            print(f"✓ Classification rate: {100 * classification_count / frame_count:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    try:
        if not MEDIAPIPE_AVAILABLE:
            print("✗ MediaPipe is required for Phase 5")
            print("  Please install MediaPipe or use Python 3.11/3.12")
            sys.exit(1)
        
        success = run_eye_state_classification()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

