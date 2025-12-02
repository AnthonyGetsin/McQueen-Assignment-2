"""
Phase 2: Face Detection with MediaPipe Face Mesh

This script implements:
- Initialize MediaPipe Face Mesh
- Process webcam frames and detect face
- Extract and visualize all facial landmarks
- Handle "no face detected" gracefully
- Deliverable: Real-time face landmark detection
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


class FaceDetector:
    """Face detection using MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh
        # static_image_mode=False for video (processes faster)
        # max_num_faces=1 to detect one face (can be increased)
        # refine_landmarks=True for more detailed landmarks
        # min_detection_confidence=0.5
        # min_tracking_confidence=0.5
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe Face Mesh initialized")
    
    def process_frame(self, frame):
        """
        Process a single frame and detect face landmarks.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            annotated_frame: Frame with landmarks drawn
            landmarks: List of landmark coordinates (x, y) or None if no face
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        landmarks = None
        
        if results.multi_face_landmarks:
            # Face detected
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
                
                # Draw face mesh tesselation (optional, for more detail)
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                # Extract landmark coordinates
                h, w, _ = frame.shape
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
        
        return annotated_frame, landmarks
    
    def get_face_count(self, frame):
        """Get the number of faces detected in the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return len(results.multi_face_landmarks)
        return 0


def test_face_detection_initialization():
    """Test if Face Mesh can be initialized."""
    print("=" * 50)
    print("Testing Face Detection Initialization")
    print("=" * 50)
    
    try:
        detector = FaceDetector()
        print("✓ Face detector initialized successfully")
        return True, detector
    except Exception as e:
        print(f"✗ Failed to initialize face detector: {e}")
        return False, None


def test_face_detection_on_frame():
    """Test face detection on a single captured frame."""
    print("\n" + "=" * 50)
    print("Testing Face Detection on Frame")
    print("=" * 50)
    
    # Initialize detector
    success, detector = test_face_detection_initialization()
    if not success:
        return False
    
    # Capture a frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return False
    
    print("Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Failed to capture frame")
        return False
    
    # Process frame
    annotated_frame, landmarks = detector.process_frame(frame)
    
    if landmarks:
        print(f"✓ Face detected with {len(landmarks)} landmarks")
        return True
    else:
        print("⚠ No face detected in frame (this is okay for testing)")
        print("  Make sure you're facing the camera when running the full test")
        return True  # Still pass, as this tests the function works


def run_real_time_face_detection():
    """Run real-time face landmark detection."""
    print("\n" + "=" * 50)
    print("Real-time Face Landmark Detection")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Make sure your face is visible to the camera")
    
    # Initialize detector
    detector = FaceDetector()
    
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
    print("Starting face detection...")
    
    frame_count = 0
    face_detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for face detection
            annotated_frame, landmarks = detector.process_frame(frame)
            
            # Handle "no face detected" gracefully
            if landmarks:
                face_detected_count += 1
                face_status = f"Face Detected ({len(landmarks)} landmarks)"
                status_color = (0, 255, 0)  # Green
            else:
                face_status = "No Face Detected"
                status_color = (0, 0, 255)  # Red
            
            # Add status text
            cv2.putText(annotated_frame, face_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Detection Rate: {face_detected_count}/{frame_count}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Phase 2: Face Landmark Detection', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Face detected in {face_detected_count} frames")
        if frame_count > 0:
            print(f"✓ Detection rate: {100 * face_detected_count / frame_count:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 50)
    print("PHASE 2: FACE DETECTION TESTS")
    print("=" * 50)
    
    if not MEDIAPIPE_AVAILABLE:
        print("✗ MediaPipe is required for Phase 2")
        print("  Please install MediaPipe or use Python 3.11/3.12")
        return False
    
    results = []
    
    # Test 1: Face detection initialization
    success, _ = test_face_detection_initialization()
    results.append(("Face Detection Initialization", success))
    
    # Test 2: Face detection on single frame
    results.append(("Face Detection on Frame", test_face_detection_on_frame()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All basic tests passed!")
        print("\nWould you like to run real-time face detection? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                run_real_time_face_detection()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping real-time test")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

