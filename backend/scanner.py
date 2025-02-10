import cv2
import numpy as np

CROP = 1

def calculate_corners(contour_points):
    corners = np.zeros((4, 2), dtype="float32")
    # the top left and the bottom right corners have the smallest and the largest sums of their x and y
    sum_points = contour_points.sum(axis=1)
    corners[0] = contour_points[np.argmin(sum_points)]
    corners[2] = contour_points[np.argmax(sum_points)]
    # the top right and the bottom left corners have the largest difference between their x and y (for the top right the difference is negative) 
    diff_points = np.diff(contour_points, axis=1)
    corners[1] = contour_points[np.argmin(diff_points)]
    corners[3] = contour_points[np.argmax(diff_points)]
    return corners

def transform_perspective(input_image, contour_points):
    corners = calculate_corners(contour_points)
    (top_left, top_right, bottom_right, bottom_left) = corners

    width_top = np.linalg.norm(bottom_right - bottom_left)
    width_bottom = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_right - bottom_right)
    height_right = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_left), int(height_right))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, destination)
    output_image = cv2.warpPerspective(input_image, matrix, (max_width, max_height))

    return output_image

def scan(img):
    img_height, img_width = img.shape[:2]

    # crop a 1.6:1 aspect ratio rectangle from the center of the image
    crop_width = round(img_width * CROP)
    crop_height = round(crop_width // 1.6)
    assert crop_height <= img_height, "Crop height exceeds image height"
    crop_y = (img_height - crop_height) // 2
    crop_x = (img_width - crop_width) // 2
    cropped_img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    # resize the cropped image to 640x400
    resized_img = cv2.resize(cropped_img, (640, 400))

    kernel = np.ones((5,5),np.uint8)
    morph_close_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, kernel, iterations= 3)

    # Find contours
    blurred_img = cv2.GaussianBlur(morph_close_img, (11, 11), 0)
    canny = cv2.Canny(blurred_img, 0, 30)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    rectangle = None
    for c in largest_contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            rectangle = c
            break

    hull = None
    transformed_image = None
    corners = None
    if rectangle is not None:
        hull = cv2.convexHull(rectangle)
        hull = hull.reshape(-1, 2)
        corners = calculate_corners(hull)
        # multiply the corners by the ratio of the original image and the resized image
        corners = corners * [crop_width / 640, crop_height / 400]
        transformed_image = transform_perspective(cropped_img, corners)

    return corners, transformed_image