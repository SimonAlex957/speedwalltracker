import cv2
import numpy as np
import sys

def detect_red_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    print("Image shape:", image.shape)
    if image is None:
        print("Error: Could not load image.")
        return None, None

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red has two HSV ranges - more sensitive
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 120, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_big_holds = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 50:  # Filter small noise
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        
        print(f"Object {i}: Center at ({center_x}, {center_y})")
        detected_big_holds.append([center_x, center_y])

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

    detected_big_holds = np.array(detected_big_holds, dtype=np.float32)

    # invert y axis to match reference coordinate system
    if detected_big_holds.size > 0:
        detected_big_holds[:,1] *= -1.0

    # Show result
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_big_holds, image


def calibrate_camera_full(known_holds, camera_holds, image_size):
    """
    Perform full camera calibration using known points
    
    Args:
        known_holds: World coordinates (X, Y, Z) with Z=0
        camera_holds: Image coordinates (x, y)
        image_size: (width, height) of image
    
    Returns:
        camera_matrix, dist_coeffs, rvecs, tvecs
    """
    # Prepare calibration data
    object_points = []  # 3D points in world coordinates
    image_points = []   # 2D points in image coordinates
    
    # Create a single array with all points (not separate arrays per point)
    obj_pts = np.zeros((len(known_holds), 1, 3), dtype=np.float32)
    img_pts = np.zeros((len(camera_holds), 1, 2), dtype=np.float32)
    
    for i in range(len(camera_holds)):
        obj_pts[i, 0] = [known_holds[i][0], known_holds[i][1], 0]  # Z=0 for floor points
        img_pts[i, 0] = [camera_holds[i][0], -camera_holds[i][1]]  # Remove y inversion for calibration
    
    object_points.append(obj_pts)
    image_points.append(img_pts)
    
    # Initial camera matrix estimation
    focal_length = image_size[1]  # Approximate focal length
    center = (image_size[0]/2, image_size[1]/2)
    initial_camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print("\nInitial camera matrix estimate:")
    print(initial_camera_matrix)
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        initial_camera_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    
    print(f"\nCalibration RMS error: {ret}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def draw_5m_line_with_homography(image, homography_matrix):
    """
    Draw a 5-meter long line in the middle of the image using the homography matrix
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Find the center of the image in pixel coordinates
    center_x_pixel = w // 2
    center_y_pixel = h // 2
    
    # Convert the center point to world coordinates using inverse homography
    # First, create a point in pixel coordinates (homogeneous)
    pixel_point = np.array([[center_x_pixel, center_y_pixel]], dtype=np.float32)
    pixel_point = pixel_point.reshape(-1, 1, 2)
    
    # Transform to world coordinates using inverse homography
    inv_homography = np.linalg.inv(homography_matrix)
    world_point = cv2.perspectiveTransform(pixel_point, inv_homography)
    world_x, world_y = world_point[0, 0]
    
    print(f"\n[Homography Method] Center pixel ({center_x_pixel}, {center_y_pixel}) maps to world coordinates ({world_x:.2f}, {world_y:.2f}) mm")
    
    # Create a 5-meter long line in world coordinates (5000 mm)
    # We'll create a horizontal line (along X axis) centered at the world point
    line_length_mm = 5000  # 5 meters = 5000 mm
    
    # Line endpoints in world coordinates (centered)
    world_start = np.array([[world_x - line_length_mm/2, world_y]], dtype=np.float32)
    world_end = np.array([[world_x + line_length_mm/2, world_y]], dtype=np.float32)
    
    # Reshape for perspectiveTransform
    world_start = world_start.reshape(-1, 1, 2)
    world_end = world_end.reshape(-1, 1, 2)
    
    # Transform world points back to image coordinates using homography
    pixel_start = cv2.perspectiveTransform(world_start, homography_matrix)
    pixel_end = cv2.perspectiveTransform(world_end, homography_matrix)
    
    # Get the pixel coordinates
    x1, y1 = int(pixel_start[0, 0, 0]), int(pixel_start[0, 0, 1])
    x2, y2 = int(pixel_end[0, 0, 0]), int(pixel_end[0, 0, 1])
    
    # Fix y-axis inversion (since we inverted y earlier)
    y1 = -y1
    y2 = -y2
    
    print(f"[Homography Method] Line in image: from ({x1}, {y1}) to ({x2}, {y2})")
    
    # Draw the line on the image
    result_image = image.copy()
    cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow line
    cv2.circle(result_image, (center_x_pixel, center_y_pixel), 8, (255, 0, 255), -1)  # Magenta center point
    
    # Add text labels
    cv2.putText(result_image, "5m line (Homography)", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result_image, "Center", (center_x_pixel+10, center_y_pixel-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return result_image


def draw_5m_line_with_calibration(image, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Draw a 5-meter long line using full camera calibration
    This method accounts for lens distortion
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Find the center of the image in pixel coordinates
    center_x_pixel = w // 2
    center_y_pixel = h // 2
    
    # We'll use the first pose (rvecs[0], tvecs[0]) for projection
    # This assumes the camera pose is consistent across all points
    
    # Create a 5-meter line in world coordinates (centered at origin of camera coordinate system)
    # But we need to transform to camera coordinates first
    
    # For simplicity, we'll create points in world coordinates and project them
    line_length_mm = 5000
    
    # Create points in world coordinates (assuming floor plane Z=0)
    # We need to find a point in world coordinates that projects to the image center
    # This is more complex with full calibration, so we'll use the inverse approach
    
    # Alternative: Create 3D points in world coordinates and project them
    world_points_3d = np.array([
        [-line_length_mm/2, 0, 0],  # Left endpoint (assuming camera looks along Z)
        [line_length_mm/2, 0, 0],   # Right endpoint
    ], dtype=np.float32).reshape(-1, 1, 3)
    
    # Project 3D points to image using the first pose
    img_points, _ = cv2.projectPoints(world_points_3d, rvecs[0], tvecs[0], 
                                      camera_matrix, dist_coeffs)
    
    x1, y1 = int(img_points[0, 0, 0]), int(img_points[0, 0, 1])
    x2, y2 = int(img_points[1, 0, 0]), int(img_points[1, 0, 1])
    
    print(f"\n[Calibration Method] Line in image: from ({x1}, {y1}) to ({x2}, {y2})")
    
    # Draw the line on the image
    result_image = image.copy()
    cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green line
    cv2.circle(result_image, (center_x_pixel, center_y_pixel), 8, (255, 0, 0), -1)  # Blue center point
    
    # Add text labels
    cv2.putText(result_image, "5m line (Full Calibration)", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_image, "Center", (center_x_pixel+10, center_y_pixel-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return result_image


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Remove lens distortion from image
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image (optional)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted


# Main execution
if len(sys.argv) < 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

# Position of known holds in millimeters
known_holds = np.array([
    (2250, 1687.5), (2375, 1937.5), (1625, 2687.5), (875, 3562.5),
    (1375, 4312.5), (1750, 4812.5), (1375, 5562.5), (1875, 6437.5),
    (2125, 7187.5), (1000, 7812.5), (1250, 8437.5), (750, 8687.5),
    (1375, 9562.5), (875, 10187.5), (1250, 10687.5), (1125, 10937.5),
    (375, 11562.5), (1375, 13312.5), (1625, 12312.5), (2125, 12937.5)
], dtype=np.float32)

# Detect red objects
camera_holds, original_image = detect_red_objects(sys.argv[1])

print("\nKnown holds (world coordinates):")
print(known_holds)
print("\nDetected holds (camera coordinates):")
print(camera_holds)

# Get image dimensions
h, w = original_image.shape[:2]
image_size = (w, h)

# Check if we have enough points
if len(camera_holds) < 4:
    print("Error: Need at least 4 detected points for homography")
    sys.exit(1)

# Method 1: Calculate homography matrix
homography_matrix, mask = cv2.findHomography(
    known_holds,
    camera_holds,
    cv2.RANSAC,
    5.0
)

print("\n" + "="*50)
print("METHOD 1: Homography Matrix")
print("="*50)
print(homography_matrix)

# Draw 5m line using homography
if homography_matrix is not None:
    result_homography = draw_5m_line_with_homography(original_image, homography_matrix)
    cv2.imwrite("image_with_5m_line_homography.jpg", result_homography)
    print("\nSaved homography result as 'image_with_5m_line_homography.jpg'")

# Method 2: Full camera calibration (if enough points)
print("\n" + "="*50)
print("METHOD 2: Full Camera Calibration")
print("="*50)

if len(camera_holds) >= 6:
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_full(
        known_holds, camera_holds, image_size)
    
    # Draw 5m line using full calibration
    result_calibration = draw_5m_line_with_calibration(
        original_image, camera_matrix, dist_coeffs, rvecs, tvecs)
    cv2.imwrite("image_with_5m_line_calibration.jpg", result_calibration)
    print("\nSaved calibration result as 'image_with_5m_line_calibration.jpg'")
    
    # Also save undistorted image
    undistorted = undistort_image(original_image, camera_matrix, dist_coeffs)
    cv2.imwrite("undistorted_image.jpg", undistorted)
    print("Saved undistorted image as 'undistorted_image.jpg'")
    
    # Create comparison image
    comparison = np.hstack((result_homography, result_calibration))
    cv2.imwrite("comparison_homography_vs_calibration.jpg", comparison)
    print("Saved comparison as 'comparison_homography_vs_calibration.jpg'")
    
    # Display both results
    cv2.imshow("Homography Method (Yellow)", result_homography)
    cv2.imshow("Full Calibration Method (Green)", result_calibration)
    cv2.imshow("Comparison", comparison)
else:
    print(f"Need at least 6 points for full calibration (have {len(camera_holds)})")
    cv2.imshow("Homography Method Only", result_homography)

cv2.waitKey(0)
cv2.destroyAllWindows()