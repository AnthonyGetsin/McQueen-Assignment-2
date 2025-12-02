import cv2
import numpy as np
import mediapipe as mp
import time


class EyeTracker:
    """Eye tracking using MediaPipe Face Mesh and EAR calculation."""
    
    def __init__(self):
        # MediaPipe Face Mesh solution for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        # Drawing utilities for visualizing landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        # Predefined drawing styles for face mesh visualization
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize FaceMesh detector
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # EAR threshold: below this value = eye closed/blink detected
        self.EAR_THRESHOLD = 0.20
        
        # MediaPipe Face Mesh uses 468 landmark points
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Temporal smoothing: require consecutive frames below threshold
        self.blink_frames = []  # Track recent frames for blink detection
        self.BLINK_FRAME_THRESHOLD = 2  # Require 2 consecutive frames below threshold
        
        # Blink counting
        self.blink_count = 0
        self.was_blinking = False  # Track previous blink state to count transitions
        
        # Blink duration tracking
        self.blink_start_time = None  # Timestamp when blink started
        self.blink_durations = []  # List of all completed blink durations (in seconds)
        self.current_blink_duration = 0.0  # Current blink duration (real-time)
        self.fps = 30  # Will be updated when camera is initialized
        
        # Frame tracking
        self.frame_count = 0
        self.last_status = None  # Track status changes for console output
        self.last_blink_status = False  # Track blink status for console updates
        
    def calculate_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio (EAR).
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        where p1-p6 are the 6 eye landmark points
        
        Args:
            eye_landmarks: List of 6 (x, y) coordinates for eye landmarks
        Returns:
            float: EAR value
        """
        # EAR requires exactly 6 points (outer, top, top-mid, inner, bottom-mid, bottom)
        if len(eye_landmarks) != 6:
            return 0.0

        points = np.array(eye_landmarks)
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])

        if horizontal == 0:
            return 0.0
        
        # EAR calculation: when eye closes, vertical distances decrease, EAR approaches 0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
        
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """
        Extract specific eye landmarks.
        
        Args:
            landmarks: MediaPipe landmark list
            indices: List of landmark indices to extract
            frame_w: Frame width
            frame_h: Frame height
        
        Returns:
            List of (x, y) coordinates
        """
        # Extract eye landmark coordinates from MediaPipe results
        eye_points = []
        for idx in indices:
            # MediaPipe landmarks are normalized (0.0-1.0), convert to pixel coordinates
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            eye_points.append((x, y))
        return eye_points
        
    def process_frame(self, frame):
        """
        Process single frame for face and eye detection.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            annotated_frame: Frame with landmarks and annotations
        """
        # MediaPipe requires RGB format, OpenCV uses BGR by default
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run face mesh detection on the frame
        results = self.face_mesh.process(rgb_frame)
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        self.frame_count += 1
        
        # Handle case where no face is detected in the frame
        if not results.multi_face_landmarks:
            # Display warning message on frame (red text)
            cv2.putText(annotated_frame, "Face Not Visible", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Blinks: {self.blink_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show average blink duration if available
            if len(self.blink_durations) > 0:
                avg_dur_ms = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                cv2.putText(annotated_frame, f"Avg Duration: {avg_dur_ms:.1f}ms", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Console status message
            duration_text = ""
            if len(self.blink_durations) > 0:
                avg_dur_ms = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                duration_text = f" | Avg Duration: {avg_dur_ms:.1f}ms"
            current_status = f"[Frame {self.frame_count}] Face Not Visible - Blinks: {self.blink_count}{duration_text}"
            if current_status != self.last_status:
                print(current_status)
                self.last_status = current_status
            
            # Reset blink state when face is not visible
            self.was_blinking = False
            self.blink_frames = []
            self.last_blink_status = False
            self.blink_start_time = None
            self.current_blink_duration = 0.0
            return annotated_frame
        
        # Process each detected face (max_num_faces=1, so typically one face)
        for face_landmarks in results.multi_face_landmarks:
            # Draw only eye meshes (left and right eye connections)
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            
            # Extract 6 landmark points for each eye (converted to pixel coordinates)
            left_eye = self.get_eye_landmarks(face_landmarks.landmark, 
                                             self.LEFT_EYE_INDICES, w, h)
            right_eye = self.get_eye_landmarks(face_landmarks.landmark,
                                              self.RIGHT_EYE_INDICES, w, h)
            
            # Calculate EAR (Eye Aspect Ratio) for each eye
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Validate that both eyes are being tracked properly
            # If EAR is 0, it likely means landmarks weren't detected properly
            valid_detection = left_ear > 0 and right_ear > 0
            
            # Check if eyes are covered (face detected but eyes not visible)
            eyes_covered = not valid_detection
            
            # Draw eye landmark points as green circles on the frame (only if visible)
            if valid_detection:
                for point in left_eye + right_eye:
                    cv2.circle(annotated_frame, point, 2, (0, 255, 0), -1)
            
            # Display EAR values with different colors
            # Frame number at the top
            cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Left EAR in cyan
            if valid_detection:
                cv2.putText(annotated_frame, f"Left EAR: {left_ear:.3f}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # Right EAR in magenta
                cv2.putText(annotated_frame, f"Right EAR: {right_ear:.3f}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                # Average EAR in yellow
                cv2.putText(annotated_frame, f"Avg EAR: {avg_ear:.3f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Handle case where eyes are covered or not visible
            if eyes_covered:
                cv2.putText(annotated_frame, "Face Not Fully Visible", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(annotated_frame, f"Blinks: {self.blink_count}", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show average blink duration if available
                if len(self.blink_durations) > 0:
                    avg_dur_ms = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                    cv2.putText(annotated_frame, f"Avg Duration: {avg_dur_ms:.1f}ms", (10, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
                # Console status message
                duration_text = ""
                if len(self.blink_durations) > 0:
                    avg_dur_ms = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                    duration_text = f" | Avg Duration: {avg_dur_ms:.1f}ms"
                current_status = f"[Frame {self.frame_count}] Face Not Fully Visible - Blinks: {self.blink_count}{duration_text}"
                if current_status != self.last_status:
                    print(current_status)
                    self.last_status = current_status
                
                # Reset blink state when eyes are covered
                self.was_blinking = False
                self.blink_frames = []
                self.last_blink_status = False
                self.blink_start_time = None
                self.current_blink_duration = 0.0
                return annotated_frame
            
            # Temporal smoothing for blink detection
            # Only detect blink if valid detection AND below threshold
            if valid_detection and avg_ear < self.EAR_THRESHOLD:
                self.blink_frames.append(True)
            else:
                self.blink_frames.append(False)
            
            # Keep only recent frames (sliding window)
            if len(self.blink_frames) > self.BLINK_FRAME_THRESHOLD:
                self.blink_frames.pop(0)
            
            # Blink detected only if we have enough consecutive frames below threshold
            is_blinking = (len(self.blink_frames) >= self.BLINK_FRAME_THRESHOLD and 
                          all(self.blink_frames))
            
            # Track blink start time and duration
            current_time = time.time()
            
            # Blink just started (transition from open to closed)
            if is_blinking and not self.was_blinking:
                self.blink_start_time = current_time
                self.blink_count += 1
                # Console message when blink is detected
                print(f"[Frame {self.frame_count}] ✓ BLINK STARTED! Total Blinks: {self.blink_count}")
            
            # Blink just ended (transition from closed to open)
            elif not is_blinking and self.was_blinking:
                if self.blink_start_time is not None:
                    blink_duration = current_time - self.blink_start_time
                    self.blink_durations.append(blink_duration)
                    self.blink_start_time = None
                    self.current_blink_duration = 0.0
                    avg_duration = sum(self.blink_durations) / len(self.blink_durations)
                    print(f"[Frame {self.frame_count}] ✓ BLINK ENDED! Duration: {blink_duration*1000:.1f}ms | Avg: {avg_duration*1000:.1f}ms")
            
            # Currently blinking - update current duration
            elif is_blinking and self.was_blinking:
                if self.blink_start_time is not None:
                    self.current_blink_duration = current_time - self.blink_start_time
            
            self.was_blinking = is_blinking
            
            # Display blink status and count with different colors
            if is_blinking:
                # Red text for blink detected
                cv2.putText(annotated_frame, "BLINK DETECTED!", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Display current blink duration in real-time
                if self.blink_start_time is not None:
                    duration_ms = self.current_blink_duration * 1000
                    cv2.putText(annotated_frame, f"Duration: {duration_ms:.1f}ms", (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Green text for eyes open
                cv2.putText(annotated_frame, "Eyes Open", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display blink count in white
            cv2.putText(annotated_frame, f"Blinks: {self.blink_count}", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display average blink duration if we have data
            if len(self.blink_durations) > 0:
                avg_duration_ms = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                cv2.putText(annotated_frame, f"Avg Duration: {avg_duration_ms:.1f}ms", (10, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Display EAR threshold for reference
            cv2.putText(annotated_frame, f"Threshold: {self.EAR_THRESHOLD}", (10, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Console status message (print every 30 frames or on status change)
            if valid_detection:
                if self.frame_count % 30 == 0 or (is_blinking != self.last_blink_status):
                    status_icon = "👁️ CLOSED" if is_blinking else "👁️ OPEN"
                    duration_info = ""
                    if is_blinking and self.blink_start_time is not None:
                        duration_info = f" | Duration: {self.current_blink_duration*1000:.1f}ms"
                    elif len(self.blink_durations) > 0:
                        avg_dur = (sum(self.blink_durations) / len(self.blink_durations)) * 1000
                        duration_info = f" | Avg Duration: {avg_dur:.1f}ms"
                    print(f"[Frame {self.frame_count}] {status_icon} | L_EAR: {left_ear:.3f} | R_EAR: {right_ear:.3f} | Avg: {avg_ear:.3f} | Blinks: {self.blink_count}{duration_info}")
                    self.last_blink_status = is_blinking
        
        return annotated_frame
        
    def run(self):
        """
        Main loop: capture, process, display.
        Handle keyboard input.
        """
        # Initialize video capture from default camera (index 0)
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Get camera resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30  # Default to 30 if FPS is not available
        
        print("=" * 60)
        print("🎯 Eye Tracker Started Successfully")
        print("=" * 60)
        print(f"📹 Camera Resolution: {width}x{height} @ {self.fps} FPS")
        print(f"⚙️  EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"📊 Blink Frame Threshold: {self.BLINK_FRAME_THRESHOLD} frames")
        print("=" * 60)
        print("💡 Press 'q' to quit")
        print("=" * 60)
        print("\n📝 Status Messages:\n")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = cap.read()
                # Check if frame was read successfully
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Process frame: detect face, extract eye landmarks, calculate EAR, detect blinks
                annotated_frame = self.process_frame(frame)
                
                # Display annotated frame in window
                cv2.imshow('Eye Tracker', annotated_frame)
                
                # Check for 'q' key press to quit (waitKey returns after 1ms)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n⚠️  Interrupted by user")
        finally:
            # Cleanup: release camera and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
            print("\n" + "=" * 60)
            print("🛑 Eye Tracker Stopped")
            print("=" * 60)
            print(f"📊 Total Frames Processed: {self.frame_count}")
            print(f"👁️  Total Blinks Detected: {self.blink_count}")
            if self.frame_count > 0:
                print(f"📈 Blinks per 100 frames: {(self.blink_count / self.frame_count * 100):.2f}")
            
            # Blink duration statistics
            if len(self.blink_durations) > 0:
                durations_ms = [d * 1000 for d in self.blink_durations]
                avg_duration = sum(durations_ms) / len(durations_ms)
                min_duration = min(durations_ms)
                max_duration = max(durations_ms)
                print(f"\n⏱️  Blink Duration Statistics:")
                print(f"   Average: {avg_duration:.1f}ms")
                print(f"   Minimum: {min_duration:.1f}ms")
                print(f"   Maximum: {max_duration:.1f}ms")
                print(f"   Total measurements: {len(self.blink_durations)}")
            else:
                print(f"\n⏱️  Blink Duration: No complete blinks recorded")
            
            print("=" * 60)


if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()
