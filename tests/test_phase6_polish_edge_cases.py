"""
Phase 6: Polish & Edge Cases

This script implements:
- Handle multiple faces (process only one)
- Add error handling for camera failures
- Implement clean exit (press 'q' to quit)
- Optimize performance if needed
- Write documentation
- Deliverable: Production-ready code with documentation
"""

import cv2
import sys
import numpy as np
import time

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


class ProductionEyeTracker:
    """
    Production-ready eye tracker with edge case handling and optimizations.
    
    Features:
    - Handles multiple faces (processes only the first one)
    - Error handling for camera failures
    - Clean exit on 'q' key press
    - Performance optimizations
    - Comprehensive status messages
    """
    
    def __init__(self, ear_threshold=0.21, camera_index=0, target_fps=30):
        """
        Initialize the eye tracker.
        
        Args:
            ear_threshold: Threshold for eye state classification
            camera_index: Camera device index (default: 0)
            target_fps: Target frames per second (for performance optimization)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required but not available. "
                            "Please install it or use Python 3.11/3.12.")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh
        # max_num_faces=1 ensures we only process one face (handles multiple faces)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Only process first face (edge case handling)
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # EAR threshold
        self.EAR_THRESHOLD = ear_threshold
        
        # Camera settings
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Statistics
        self.frame_count = 0
        self.faces_detected_count = 0
        self.blink_count = 0
        self.start_time = None
        
        print("✓ Production Eye Tracker initialized")
        print(f"  - EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"  - Camera Index: {self.camera_index}")
        print(f"  - Target FPS: {self.target_fps}")
    
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
        """Classify eye state based on EAR threshold."""
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
    
    def process_frame(self, frame, previous_state=None):
        """
        Process a single frame with error handling.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            previous_state: Previous eye state for blink detection
        
        Returns:
            annotated_frame: Frame with annotations
            left_state: Left eye state or None
            right_state: Right eye state or None
            avg_ear: Average EAR or None
            current_state: Overall state for blink detection
        """
        try:
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            annotated_frame = frame.copy()
            h, w, _ = frame.shape
            left_state = None
            right_state = None
            avg_ear = None
            current_state = None
            
            # Handle multiple faces - only process first face (max_num_faces=1 handles this)
            if results.multi_face_landmarks:
                self.faces_detected_count += 1
                
                # Get first face only (edge case: multiple faces)
                face_landmarks = results.multi_face_landmarks[0]
                
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
                
                # Determine overall state
                if left_state == "CLOSED" and right_state == "CLOSED":
                    current_state = "CLOSED"
                else:
                    current_state = "OPEN"
                
                # Detect blinks (transition from OPEN to CLOSED)
                if previous_state == "OPEN" and current_state == "CLOSED":
                    self.blink_count += 1
                
                # Draw eye landmarks with color coding
                left_color = (0, 255, 0) if left_state == "OPEN" else (0, 0, 255)
                right_color = (0, 255, 0) if right_state == "OPEN" else (0, 0, 255)
                
                for point in left_eye:
                    cv2.circle(annotated_frame, point, 3, left_color, -1)
                for point in right_eye:
                    cv2.circle(annotated_frame, point, 3, right_color, -1)
            
            return annotated_frame, left_state, right_state, avg_ear, current_state
            
        except Exception as e:
            print(f"⚠ Error processing frame: {e}")
            return frame, None, None, None, None
    
    def run(self):
        """
        Main loop with error handling and clean exit.
        
        Handles:
        - Camera initialization failures
        - Frame read failures
        - Keyboard interrupts
        - Clean resource cleanup
        """
        # Initialize camera with error handling
        print(f"\nAttempting to open camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_index}")
            print("  Please check:")
            print("  - Camera is connected")
            print("  - Camera permissions are granted")
            print("  - No other application is using the camera")
            return False
        
        print("✓ Camera opened successfully")
        
        # Set camera properties (if supported)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera: {width}x{height} @ {fps} FPS")
        print("\n" + "=" * 50)
        print("Eye Tracker Running")
        print("=" * 50)
        print("Press 'q' to quit")
        print("Make sure your face is visible to the camera")
        print("Try blinking to see the classification change!")
        print("=" * 50 + "\n")
        
        self.start_time = time.time()
        previous_state = None
        last_frame_time = time.time()
        
        try:
            while True:
                # Performance optimization: control frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
                last_frame_time = time.time()
                
                # Read frame with error handling
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Warning: Failed to read frame from camera")
                    print("  Attempting to continue...")
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                
                self.frame_count += 1
                
                # Process frame
                annotated_frame, left_state, right_state, avg_ear, current_state = \
                    self.process_frame(frame, previous_state)
                
                if current_state is not None:
                    previous_state = current_state
                
                # Display information
                y_offset = 30
                
                if left_state is not None and right_state is not None and avg_ear is not None:
                    # Determine overall state
                    if left_state == "CLOSED" and right_state == "CLOSED":
                        overall_state = "CLOSED"
                        state_color = (0, 0, 255)  # Red
                    else:
                        overall_state = "OPEN"
                        state_color = (0, 255, 0)  # Green
                    
                    # Display overall state
                    cv2.putText(annotated_frame, f"EYE STATE: {overall_state}",
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 3)
                    y_offset += 50
                    
                    # Display individual eye states
                    left_color = (0, 255, 0) if left_state == "OPEN" else (0, 0, 255)
                    right_color = (0, 255, 0) if right_state == "OPEN" else (0, 0, 255)
                    
                    cv2.putText(annotated_frame, f"Left: {left_state}",
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
                    y_offset += 35
                    
                    cv2.putText(annotated_frame, f"Right: {right_state}",
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
                    y_offset += 35
                    
                    # Display EAR value
                    cv2.putText(annotated_frame, f"EAR: {avg_ear:.3f}",
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 30
                    
                    # Display blink count
                    cv2.putText(annotated_frame, f"Blinks: {self.blink_count}",
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 35
                    
                    status = "Face Detected"
                    status_color = (0, 255, 0)  # Green
                else:
                    status = "No Face Detected"
                    status_color = (0, 0, 255)  # Red
                
                cv2.putText(annotated_frame, status, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
                y_offset += 30
                
                # Display statistics
                elapsed_time = time.time() - self.start_time
                actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(annotated_frame, f"Frame: {self.frame_count}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                cv2.putText(annotated_frame, f"FPS: {actual_fps:.1f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                detection_rate = (self.faces_detected_count / self.frame_count * 100) \
                    if self.frame_count > 0 else 0
                cv2.putText(annotated_frame, f"Detection: {detection_rate:.1f}%",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display quit instruction
                cv2.putText(annotated_frame, "Press 'q' to quit",
                           (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Phase 6: Production Eye Tracker', annotated_frame)
                
                # Handle keyboard input - clean exit on 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' key pressed - exiting...")
                    break
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received - exiting...")
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean resource cleanup
            print("\nCleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\n" + "=" * 50)
            print("Session Statistics")
            print("=" * 50)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Faces detected: {self.faces_detected_count}")
            print(f"Blinks detected: {self.blink_count}")
            print(f"Detection rate: {(self.faces_detected_count / self.frame_count * 100) if self.frame_count > 0 else 0:.1f}%")
            print(f"Average FPS: {actual_fps:.1f}")
            print(f"Total runtime: {elapsed_time:.1f} seconds")
            print("=" * 50)
            print("✓ Eye Tracker stopped successfully")
        
        return True


def main():
    """Main entry point with command-line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production Eye Tracker - Phase 6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase6_polish_edge_cases.py
  python phase6_polish_edge_cases.py --threshold 0.25
  python phase6_polish_edge_cases.py --camera 1 --fps 60
        """
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.21,
        help='EAR threshold for classification (default: 0.21)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second (default: 30)'
    )
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("✗ MediaPipe is required for Phase 6")
        print("  Please install MediaPipe or use Python 3.11/3.12")
        return 1
    
    try:
        tracker = ProductionEyeTracker(
            ear_threshold=args.threshold,
            camera_index=args.camera,
            target_fps=args.fps
        )
        success = tracker.run()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

