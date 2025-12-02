"""
Phase 3: Eye Landmark Extraction

This script implements:
- Identify correct eye landmark indices from MediaPipe documentation
- Extract 6 landmarks per eye (12 total)
- Convert normalized coordinates to pixel coordinates
- Draw eye contours on video feed
- Deliverable: Visual eye contour overlay
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


class EyeLandmarkExtractor:
    """Extract and visualize eye landmarks from MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh."""
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
        
        # Eye landmark indices (MediaPipe Face Mesh has 468 landmarks)
        # According to project requirements:
        # Left Eye: Indices [362, 385, 387, 263, 373, 380]
        # Right Eye: Indices [33, 160, 158, 133, 153, 144]
        # Note: These are from the user's perspective (left/right as user sees them)
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        print("✓ Eye Landmark Extractor initialized")
        print(f"  - Left eye indices: {self.LEFT_EYE_INDICES}")
        print(f"  - Right eye indices: {self.RIGHT_EYE_INDICES}")
    
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """
        Extract specific eye landmarks and convert to pixel coordinates.
        
        MediaPipe returns normalized coordinates (0.0-1.0), so we need to
        multiply by frame width/height to get pixel coordinates.
        
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
            # Convert normalized coordinates (0.0-1.0) to pixel coordinates
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            eye_points.append((x, y))
        return eye_points
    
    def draw_eye_contour(self, frame, eye_points, color=(0, 255, 0)):
        """
        Draw eye contour by connecting the 6 landmark points.
        
        Args:
            frame: Frame to draw on
            eye_points: List of 6 (x, y) coordinates
            color: BGR color tuple for the contour
        """
        if len(eye_points) != 6:
            return
        
        # Draw points
        for point in eye_points:
            cv2.circle(frame, point, 3, color, -1)
        
        # Draw contour by connecting points
        # Connect points in order: p1 -> p2 -> p3 -> p4 -> p5 -> p6 -> p1
        points = np.array(eye_points, dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        
        # Also draw lines between specific points for better visualization
        # Horizontal line (p1 to p4)
        cv2.line(frame, eye_points[0], eye_points[3], color, 1)
        # Vertical lines
        cv2.line(frame, eye_points[1], eye_points[5], color, 1)
        cv2.line(frame, eye_points[2], eye_points[4], color, 1)
    
    def process_frame(self, frame):
        """
        Process a single frame and extract eye landmarks.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            annotated_frame: Frame with eye contours drawn
            left_eye_points: List of left eye landmark coordinates or None
            right_eye_points: List of right eye landmark coordinates or None
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        left_eye_points = None
        right_eye_points = None
        
        if results.multi_face_landmarks:
            # Face detected - extract eye landmarks
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks (convert normalized to pixel coordinates)
                left_eye_points = self.get_eye_landmarks(
                    face_landmarks.landmark, 
                    self.LEFT_EYE_INDICES, 
                    w, h
                )
                right_eye_points = self.get_eye_landmarks(
                    face_landmarks.landmark,
                    self.RIGHT_EYE_INDICES,
                    w, h
                )
                
                # Draw eye contours
                self.draw_eye_contour(annotated_frame, left_eye_points, 
                                    color=(0, 255, 0))  # Green for left
                self.draw_eye_contour(annotated_frame, right_eye_points,
                                    color=(255, 0, 0))  # Blue for right
                
                # Label the eyes
                if left_eye_points:
                    cv2.putText(annotated_frame, "Left Eye", 
                              (left_eye_points[0][0] - 30, left_eye_points[0][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if right_eye_points:
                    cv2.putText(annotated_frame, "Right Eye",
                              (right_eye_points[0][0] - 30, right_eye_points[0][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return annotated_frame, left_eye_points, right_eye_points


def run_eye_landmark_extraction():
    """Run real-time eye landmark extraction."""
    print("\n" + "=" * 50)
    print("Phase 3: Eye Landmark Extraction")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Make sure your face is visible to the camera")
    
    # Initialize extractor
    try:
        extractor = EyeLandmarkExtractor()
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
    print("Starting eye landmark extraction...")
    
    frame_count = 0
    eyes_detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for eye landmark extraction
            annotated_frame, left_eye, right_eye = extractor.process_frame(frame)
            
            # Handle "no face detected" gracefully
            if left_eye and right_eye:
                eyes_detected_count += 1
                status = "Eyes Detected"
                status_color = (0, 255, 0)  # Green
            else:
                status = "No Face Detected"
                status_color = (0, 0, 255)  # Red
            
            # Add status text
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if left_eye and right_eye:
                cv2.putText(annotated_frame, 
                           f"Left Eye: {len(left_eye)} landmarks", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated_frame,
                           f"Right Eye: {len(right_eye)} landmarks", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(annotated_frame, "Press 'q' to quit", 
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Phase 3: Eye Landmark Extraction', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Eyes detected in {eyes_detected_count} frames")
        if frame_count > 0:
            print(f"✓ Detection rate: {100 * eyes_detected_count / frame_count:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    try:
        if not MEDIAPIPE_AVAILABLE:
            print("✗ MediaPipe is required for Phase 3")
            print("  Please install MediaPipe or use Python 3.11/3.12")
            sys.exit(1)
        
        success = run_eye_landmark_extraction()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

