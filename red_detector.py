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

    return image, detected_big_holds








def find_best_similarity(known_pts, camera_pts, tol=10.0, min_inliers=3):
    """
    known_pts : (N,2) numpy array – reference hold coords
    camera_pts: (M,2) numpy array – detected holds in view
    tol       : maximum distance (pixels) to consider a match
    min_inliers: minimum number of correspondences to accept the transform

    returns (M×3) affine matrix or None
    """
    # sort by y (lowest first)
    known = known_pts[np.argsort(known_pts[:,1])]
    cam   = camera_pts[np.argsort(camera_pts[:,1])]

    # try every offset as described
    for i in range(len(known)):
        for j in range(1, len(known) - i):
            src = cam[:2].astype(np.float32)      # first two camera points
            dst = known[i:i+2].astype(np.float32) # candidate pair
            if np.linalg.norm(src[1]-src[0]) < 1e-6:
                continue  # degenerate

            # estimate similarity via two-point solution
            # cv2.estimateAffinePartial2D does scale+rot+trans from ≥2 points
            M, inliers = cv2.estimateAffinePartial2D(src, dst,
                                                     method=cv2.LMEDS)  # or RANSAC
            if M is None:
                continue

            # transform all camera points
            cam_hom = np.hstack([cam, np.ones((len(cam),1))])
            transformed = (M @ cam_hom.T).T           # Nx2

            # count how many are near a known point
            dists = np.linalg.norm(transformed[:,None,:] - known[None,:,:], axis=2)
            best = np.min(dists, axis=1)
            inlier_count = np.count_nonzero(best < tol)

            if inlier_count >= min_inliers:
                return M  # good transform found
    return None








def calibrate_from_three(camera_pts, known_pts):
    """Compute affine transform from first three sorted points.

    camera_pts and known_pts are expected as Nx2 arrays.  Points are
    sorted by increasing y, then the first three of each list are used
    as correspondences.  Returns 2x3 matrix mapping camera->known or
    None if computation fails.
    """
    if camera_pts.shape[0] < 3 or known_pts.shape[0] < 3:
        return None

    cam_sorted = camera_pts[np.argsort(camera_pts[:,1])].astype(np.float32)
    known_sorted = known_pts[np.argsort(known_pts[:,1])].astype(np.float32)

    src = cam_sorted[:3]
    dst = known_sorted[:3]

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M





def camera_to_cartesian(camera_pts, transform):
    """Apply affine `transform` (2x3) to camera_pts, returning Cartesian coords.

    `camera_pts` should be an (N,2) array of pixel coordinates, and
    `transform` is the 2x3 matrix produced by one of the calibration
    routines.  The output is an (N,2) array in the known/cartesian
    reference frame.
    """
    if transform is None or camera_pts is None:
        return None
    pts = np.array(camera_pts, dtype=np.float32)
    hom = np.hstack([pts, np.ones((len(pts),1))])
    return (transform @ hom.T).T

# usage (example)
known_holds = np.array([
    (2250, 1687.5), (2375, 1937.5), (1625, 2687.5), (875, 3562.5),
    (1375, 4312.5), (1750, 4812.5), (1375, 5562.5), (1875, 6437.5),
    (2125, 7187.5), (1000, 7812.5), (1250, 8437.5), (750, 8687.5),
    (1375, 9562.5), (875, 10187.5), (1250, 10687.5), (1125, 10937.5),
    (375, 11562.5), (1375, 13312.5), (1625, 12312.5), (2125, 12937.5)
], dtype=np.float32)   # or whatever dtype you need


   # from your detector



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 red_detector.py image.jpg")
    else:
        image, detected_big_holds = detect_red_objects(sys.argv[1])
        if detected_big_holds is None or detected_big_holds.size == 0:
            print("no holds detected, cannot compute transform")
        else:
            cam_holds = np.array(detected_big_holds, dtype=np.float32)

            # first‑3‑point calibration
            M3 = calibrate_from_three(cam_holds, known_holds)
            if M3 is not None:
                cam_hom = np.hstack([cam_holds, np.ones((len(cam_holds),1))])
                cam_in_known = (M3 @ cam_hom.T).T
                print("camera points mapped into known frame using first three holds:")
                print(cam_in_known)
                # draw predicted known positions back onto image
                H = np.vstack([M3, [0,0,1]])
                try:
                    Hinv = np.linalg.inv(H)
                    Minv = Hinv[:2,:]
                    kn_hom = np.hstack([known_holds, np.ones((len(known_holds),1))])
                    proj = (Minv @ kn_hom.T).T
                    for idx,(x,y) in enumerate(proj):
                        cv2.circle(image, (int(x), int(y)), 4, (0,0,255), -1)
                        cv2.putText(image, str(idx), (int(x)+5, int(y)+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    cv2.imwrite("output.jpg", image)
                    print("calibrated known positions plotted on output.jpg")
                except np.linalg.LinAlgError:
                    print("failed to invert calibration matrix")
            else:
                print("could not compute 3-point calibration")

            # still run full similarity if needed
            M = find_best_similarity(known_holds, cam_holds)
            if M is not None:
                scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
                angle = np.degrees(np.arctan2(M[1,0], M[0,0]))
                print("scale", scale, "rotation", angle, "deg")
            else:
                print("no consistent match")