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
        detected_big_holds[:,1] = image.shape[0] - detected_big_holds[:,1]

    # Show result
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_big_holds, image


def draw_5m_line(image, homography_matrix):
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
    
    print(f"Center pixel ({center_x_pixel}, {center_y_pixel}) maps to world coordinates ({world_x:.2f}, {world_y:.2f}) mm")
    
    # Create a 5-meter long line in world coordinates (5000 mm)
    # We'll create a vertical line (along Y axis) centered at the world point
    line_length_mm = 5000  # 5 meters = 5000 mm
    # Line endpoints in world coordinates (centered, vertical)
    world_start = np.array([[world_x, world_y - line_length_mm/2]], dtype=np.float32)
    world_end = np.array([[world_x, world_y + line_length_mm/2]], dtype=np.float32)
    
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
    y1 = image.shape[0] - y1
    y2 = image.shape[0] - y2
    
    print(f"Line in image: from ({x1}, {y1}) to ({x2}, {y2})")
    
    # Draw the line on the image
    result_image = image.copy()
    cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow line
    cv2.circle(result_image, (center_x_pixel, center_y_pixel), 8, (255, 0, 255), -1)  # Magenta center point
    
    # Add text labels
    cv2.putText(result_image, "5m line", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result_image, "Center", (center_x_pixel+10, center_y_pixel-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return result_image


# usage (example)
# possision of known holds in millimeters
known_holds = np.array([
    (2250, 1687.5), (2375, 1937.5), (1625, 2687.5), (875, 3562.5),
    (1375, 4312.5), (1750, 4812.5), (1375, 5562.5), (1875, 6437.5),
    (2125, 7187.5), (1000, 7812.5), (1250, 8437.5), (750, 8687.5),
    (1375, 9562.5), (875, 10187.5), (1250, 10687.5), (1125, 10937.5),
    (375, 11562.5), (1375, 13312.5), (1625, 12312.5), (2125, 12937.5)
], dtype=np.float32)   # or whatever dtype you need


# from your detector
camera_holds, original_image = detect_red_objects(sys.argv[1])

print("Known holds (world coordinates):")
print(known_holds)
print("\nDetected holds (camera coordinates):")
print(camera_holds)

# Calculate homography matrix
homography_matrix, mask = cv2.findHomography(
            known_holds,
            camera_holds,
            cv2.RANSAC,
            5.0
        )

print("\nHomography matrix:")
print(homography_matrix)

# Draw the 5-meter line
if homography_matrix is not None:
    result = draw_5m_line(original_image, homography_matrix)
    
    # Save and display the result
    cv2.imwrite("image_with_5m_line.jpg", result)
    print("\nSaved image with 5m line as 'image_with_5m_line.jpg'")
    
   
else:
    print("Error: Homography matrix could not be computed")
