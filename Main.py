import cv2
import json
import numpy as np
import os
import sys


if len(sys.argv) < 2:
    # Display an error message if no folder path is provided
    print("Please provide the folder path as an argument.")
    sys.exit(1)

folder_path = sys.argv[1]


def Aruco(image, intrinsic_matrix, id):  # Find ArUco marker information with identifier id
    output = image.copy()
    marker_info = next((item for item in aruco_data if item["identifier"] == id), None)
    if marker_info is None:
        raise ValueError("ArUco marker with identifier 5 not found.")

    marker_id = marker_info["identifier"]
    marker_size = marker_info["size"]  # Actual size of the marker in meters

    # Get the ArUco dictionary
    if marker_info["aruco_dict"] == "DICT_4X4_50":
        aruco_dict = cv2.aruco.DICT_4X4_50
    elif marker_info["aruco_dict"] == "DICT_6X6_250":
        aruco_dict = cv2.aruco.DICT_6X6_250
    else:
        raise ValueError(f"Unsupported ArUco dictionary: {marker_info['aruco_dict']}")

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)

    # Find the specific marker and estimate its pose
    if ids is not None and any(ids.flatten() == marker_id):
        index = np.where(ids.flatten() == marker_id)[0][0]
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[index], marker_size, intrinsic_matrix, dist_coeffs)
        # Display the image with the detected marker
        cv2.aruco.drawDetectedMarkers(output, corners)
        if len(rvec) > 0:  # If pose is estimated
            cv2.drawFrameAxes(output, intrinsic_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)
        cv2.imshow("Detection of ArUco marker", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return tvec.flatten(), rvec.flatten()
    else:
        return print("Marker ID {} not found in the image.".format(marker_id))


def detect_circles(image, circularity_threshold=0.8):  # Function to detect circles in an image
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 0, 250)
    kernel = np.ones((2, 2), np.uint8)
    # Apply dilation to edges to close them
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_points = []  # To store opposite points on circles

    # Check each contour
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity > circularity_threshold:
            # Opposite points on the circle
            point1 = np.array([center[0], center[1] - radius], dtype='float32')
            point2 = np.array([center[0], center[1] + radius], dtype='float32')

            # Draw the circle and points
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            cv2.circle(output, (center[0], center[1] - radius), 5, (255, 0, 0), -1)
            cv2.circle(output, (center[0], center[1] + radius), 5, (0, 0, 255), -1)

    # Display the image with detected circles
    cv2.imshow('Circle described by the object', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return point1, point2


def pixel_to_world(x, y, Z, intrinsic_matrix):  # Function to convert pixel coordinates to world coordinates
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Conversion of pixel coordinates to world coordinates
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return np.array([X, Y, Z])


def create_transformation_matrix(rotation_vector, translation_vector):  # Function to create a transformation matrix from a rotation and translation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Create 4x4 transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def cam2_in_cam1(rvec1, tvec1, rvec2, tvec2):  # Function to calculate the position of camera x in camera y's frame with respect to a reference.
    R1, _ = cv2.Rodrigues(rvec1)
    R1_inv = np.linalg.inv(R1)
    tvec1_inv = -np.dot(R1_inv, tvec1)

    # Apply the second transformation
    R2, _ = cv2.Rodrigues(rvec2)
    R_combined = np.dot(R2, R1_inv)
    tvec_combined = np.dot(R2, tvec1_inv) + tvec2

    # Convert combined rotation matrix to rotation vector
    rvec_combined, _ = cv2.Rodrigues(R_combined)

    return rvec_combined, tvec_combined


def project_point_to_other_camera(point, Z, intrinsic_cam1, intrinsic_cam2, rvec_cam2_in_cam1, tvec_cam2_in_cam1):  # Function to project a point from one camera to another
    # Convert the point to 3D coordinates in camera 1's reference frame
    point3D_cam1 = pixel_to_world(point[0], point[1], Z, intrinsic_cam1)
    # Convert rotation vector to rotation matrix
    R_cam2_in_cam1, _ = cv2.Rodrigues(rvec_cam2_in_cam1)
    # Apply transformation to get coordinates in camera 2's reference frame
    point3D_cam2 = np.dot(R_cam2_in_cam1, point3D_cam1.reshape(3, 1)) + tvec_cam2_in_cam1.reshape(3, 1)
    # Project the 3D point onto camera 2's image
    point2D_cam2 = np.dot(intrinsic_cam2, point3D_cam2)
    point2D_cam2 /= point2D_cam2[2]  # Normalize to get pixel coordinates
    return point2D_cam2[:2].ravel()


def segment_and_show(image, point1, point2):  # Function to segment a region of interest in an image and find the highest point
    height, width = image.shape[:2]

    x_min = int(min(point1[0], point2[0]))
    x_max = int(max(point1[0], point2[0]))

    y_min = 0
    y_max = height

    roi = image[y_min:y_max, x_min:x_max]

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary_roi = cv2.threshold(gray_roi, 75, 200, cv2.THRESH_BINARY)

    black_points = np.column_stack(np.where(binary_roi == 0))

    highest_point_cropped = black_points[np.argmin(black_points[:, 0])]


    highest_point_original = (highest_point_cropped[1] + x_min, highest_point_cropped[0] + y_min)

    cv2.circle(image, (highest_point_original[0], highest_point_original[1]), 5, (255, 0, 0), -1)

    cv2.imshow("Highest Point", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return highest_point_original


def load_files_from_folder(folder_path):  # Function to load files from a folder
    intrinsics = None
    aruco_data = None
    image1 = None
    image2 = None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
                if 'Camera1' in data:  # Assumption about data structure for intrinsics
                    intrinsics = data
                else:
                    aruco_data = data
        elif filename == 'captured_image_camera1_ex2.jpg':
            image1 = cv2.imread(file_path)
        elif filename == 'captured_image_camera2_ex2.jpg':
            image2 = cv2.imread(file_path)

    if intrinsics is None or aruco_data is None or image1 is None or image2 is None:
        raise ValueError("Necessary files were not found in the folder.")

    return intrinsics, aruco_data, image1, image2


intrinsics, aruco_data, image1, image2 = load_files_from_folder(folder_path)

camera1_params = intrinsics['Camera1']
camera1_matrix = np.array([[camera1_params['fx'], 0, camera1_params['ppx']],
                           [0, camera1_params['fy'], camera1_params['ppy']],
                           [0, 0, 1]])
camera2_params = intrinsics['Camera2']
camera2_matrix = np.array([[camera2_params['fx'], 0, camera2_params['ppx']],
                           [0, camera2_params['fy'], camera2_params['ppy']],
                           [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))  # Assume zero distortion



tcam1_5, Rcam1_5 = Aruco(image1, camera1_matrix, 5)
print("The translation of the table in camera 1's frame is: ", tcam1_5)
print("The rotation of the table in camera 1's frame is: ", Rcam1_5)



P1, P2 = detect_circles(image1, circularity_threshold=0.8)
# P1 Pixel coordinates for point 1
# P2 Pixel coordinates for point 2

Zt = tcam1_5[2]  # Table depth in meters

world_point1 = pixel_to_world(P1[0], P1[1], Zt, camera1_matrix)
world_point2 = pixel_to_world(P2[0], P2[1], Zt, camera1_matrix)

distance = np.linalg.norm(world_point2 - world_point1) * 1000
print(f"Object diameter in mm = {distance} mm")



tcam1_0, Rcam1_0 = Aruco(image1, camera1_matrix, 0)
print("The translation of Aruco 0 in camera 1's frame is: ", tcam1_0)
print("The rotation of Aruco 0 in camera 1's frame is: ", Rcam1_0)
tcam2_0, Rcam2_0 = Aruco(image2, camera2_matrix, 0)
print("The translation of Aruco 0 in camera 2's frame is: ", tcam2_0)
print("The rotation of Aruco 0 in camera 2's frame is: ", Rcam2_0)

# Camera 2 position in camera 1's frame
rvec_cam2_in_cam1, tvec_cam2_in_cam1 = cam2_in_cam1(Rcam2_0, tcam2_0, Rcam1_0, tcam1_0)
T_matrix_cam2_cam1 = create_transformation_matrix(rvec_cam2_in_cam1, tvec_cam2_in_cam1)
print("Transformation matrix of Camera 2 in Camera 1's frame: ", T_matrix_cam2_cam1)
R = T_matrix_cam2_cam1[:3, :3]
T_vec = T_matrix_cam2_cam1[:3, 3]
R_inv = R.T
T_inv = -R_inv @ T_vec
T_matrix_cam1_cam2 = np.eye(4)
T_matrix_cam1_cam2[:3, :3] = R_inv
T_matrix_cam1_cam2[:3, 3] = T_inv
print("Transformation matrix of Camera 1 in Camera 2's frame: ", T_matrix_cam1_cam2)



Z = tcam1_5[2]  # Depth of point = depth of Aruco 1

point1_in_cam2 = project_point_to_other_camera(P1, Z, camera1_matrix, camera2_matrix, cv2.Rodrigues(R_inv)[0], T_inv)
point2_in_cam2 = project_point_to_other_camera(P2, Z, camera1_matrix, camera2_matrix, cv2.Rodrigues(R_inv)[0], T_inv)

highest_point = segment_and_show(image2, point1_in_cam2, point2_in_cam2)

Zh = tcam2_0[2]  # Depth of this point is the same as depth of Aruco 0
# Projection of pixel points into world coordinates
world_point_h = pixel_to_world(highest_point[0], highest_point[1], Zh, camera2_matrix)
P_cam2 = np.append(world_point_h, 1)
p1_cam1 = np.dot(T_matrix_cam2_cam1, P_cam2)
P_cam1 = p1_cam1[:3] / p1_cam1[3]
h = (tcam1_5[2] - P_cam1[2]) * 1000
print(f"Object height in mm = {h} mm")



results = {
    'Table translation in Camera 1 frame': tcam1_5,
    'Table rotation in Camera 1 frame': Rcam1_5,
    'Projection of two opposite circle points in Camera 1 frame': [world_point1, world_point2],
    'Diameter in mm': distance,
    'Aruco 0 translation in Camera 1 frame': tcam1_0,
    'Aruco 0 rotation in Camera 1 frame': Rcam1_0,
    'Aruco 0 translation in Camera 2 frame': tcam2_0,
    'Aruco 0 rotation in Camera 2 frame': Rcam2_0,
    'Transformation matrix of Camera 2 in Camera 1 frame': T_matrix_cam2_cam1,
    'Transformation matrix of Camera 1 in Camera 2 frame': T_matrix_cam1_cam2,
    'Highest point of the object': highest_point,
    'Coordinates of highest point in Camera 2 frame': world_point_h,
    'Coordinates of highest point in Camera 1 frame': P_cam1,
    'Object height in mm': h,
}


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Path of the file where you want to save the results
result_file_path = sys.argv[1]  # In my case, it's the same folder where input files are

# Writing results to a JSON file
with open(result_file_path, 'w') as file:
    json.dump(results, file, default=convert_numpy, indent=4)

print(f"Results saved in {result_file_path}")
