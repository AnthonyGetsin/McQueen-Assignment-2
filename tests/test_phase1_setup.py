"""
Phase 1: Environment Setup Tests

This script verifies that the environment is properly set up:
- MediaPipe installation and version
- Webcam access with OpenCV
- Basic webcam display functionality
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


def test_opencv_installation():
    """Verify OpenCV is installed and importable."""
    print("=" * 50)
    print("Testing OpenCV Installation")
    print("=" * 50)
    
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("✓ OpenCV is properly installed")
        
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        if test_image.shape == (100, 100, 3):
            print("✓ OpenCV numpy integration working")
        
        return True
    except Exception as e:
        print(f"✗ OpenCV installation failed: {e}")
        return False


def test_mediapipe_installation():
    """Verify MediaPipe is installed and importable."""
    print("=" * 50)
    print("Testing MediaPipe Installation")
    print("=" * 50)
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠ MediaPipe is not available for this Python version")
        print("  Note: MediaPipe may not support Python 3.13+ yet")
        print("  Consider using Python 3.11 or 3.12 for full MediaPipe support")
        return False
    
    try:
        print(f"MediaPipe version: {mp.__version__}")
        print("✓ MediaPipe is properly installed")
        return True
    except Exception as e:
        print(f"✗ MediaPipe installation failed: {e}")
        return False


def test_webcam_access():
    """Test if webcam can be accessed and frames can be captured."""
    print("\n" + "=" * 50)
    print("Testing Webcam Access")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Failed to access webcam")
        return False
    
    print("✓ Webcam opened successfully")
    
    # Get webcam properties using OpenCV
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame captured successfully (Shape: {frame.shape})")
        # Verify frame is a valid numpy array
        if isinstance(frame, np.ndarray):
            print(f"  - Frame type: {frame.dtype}")
            print(f"  - Frame size: {frame.size} pixels")
        cap.release()
        return True
    else:
        print("✗ Failed to capture frame from webcam")
        cap.release()
        return False


def test_webcam_display():
    """Display webcam feed to verify it's working."""
    print("\n" + "=" * 50)
    print("Testing Webcam Display")
    print("=" * 50)
    print("Press 'q' to quit the webcam display")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Failed to open webcam for display")
        return False
    
    print("✓ Webcam display started")
    print("  - Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add text overlay using OpenCV to show it's working
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add OpenCV version info
        cv2.putText(frame, f"OpenCV {cv2.__version__}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame using OpenCV
        cv2.imshow('Phase 1: Webcam Test - OpenCV', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"✓ Webcam display test completed ({frame_count} frames displayed)")
        return True
    else:
        print("✗ No frames were displayed")
        return False


def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 50)
    print("PHASE 1: ENVIRONMENT SETUP TESTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: OpenCV installation
    results.append(("OpenCV Installation", test_opencv_installation()))
    
    # Test 2: MediaPipe installation
    results.append(("MediaPipe Installation", test_mediapipe_installation()))
    
    # Test 3: Webcam access
    results.append(("Webcam Access", test_webcam_access()))
    
    # Test 4: Webcam display (optional - requires user interaction)
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    # Critical tests: OpenCV and Webcam Access (MediaPipe is optional for Phase 1)
    critical_tests = [results[0], results[2]]  # OpenCV and Webcam Access
    all_critical_passed = all(result for _, result in critical_tests)
    
    if all_critical_passed:
        print("\n✓ All basic tests passed!")
        print("\nWould you like to test webcam display? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            test_webcam_display()
    else:
        print("\n✗ Some tests failed. Please check your setup.")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

