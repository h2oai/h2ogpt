import os
import numpy as np
from scipy.stats import mode
import cv2
from imutils.perspective import four_point_transform


def largest_contour(contours):
    """ Find the largest contour in the list. """
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    return largest_contour


def is_contour_acceptable(contour, image, size_threshold=0.1, aspect_ratio_range=(0.5, 2), rotation_threshold=30):
    """ Check if the contour is acceptable based on size, aspect ratio, and rotation. """
    # Size check
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(contour)
    if contour_area / image_area < size_threshold or contour_area / image_area > 1 - size_threshold:
        return False

    # Aspect ratio check
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
        return False

    # Rotation check
    _, _, angle = cv2.minAreaRect(contour)
    if angle > rotation_threshold:
        return False

    return True


def align_image(img_file):
    # Load the image
    # img_file = '/home/jon/Downloads/fastfood.jpg'
    # img_file = "/home/jon/Documents/reciept.jpg"
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest = largest_contour(contours)

    if largest is not None and is_contour_acceptable(largest, image):
        # Approximate the contour to a polygon
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        # If the approximated contour has four points, assume it is a quadrilateral
        if len(approx) == 4:
            warped = four_point_transform(image, approx.reshape(4, 2))
            out_file = img_file + "_aligned.jpg"
            cv2.imwrite(out_file, warped)
            return out_file
        else:
            print("Contour is not a quadrilateral.")
    else:
        print("No acceptable contours found.")


def correct_rotation(img_file, border_size=50):
    # Function to rotate the image to the correct orientation
    # Load the image
    image = cv2.imread(img_file)

    # Check if image is loaded
    if image is None:
        raise ValueError("Error: Image not found.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect points that form a line using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)

    # Initialize list of angles
    angles = []

    # Loop over the lines and compute the angle of each line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Calculate the most frequent angle in the image
    most_frequent_angle = mode(np.round(angles)).mode

    # Assuming the receipt is horizontal, the text should be near 0 or -180/180 degrees
    # We need to bring the angle to the range (-45, 45) to minimize rotation and keep the text upright
    if most_frequent_angle < -45:
        most_frequent_angle += 90
    elif most_frequent_angle > 45:
        most_frequent_angle -= 90

    # Rotate the original image by the most frequent angle to correct its orientation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, most_frequent_angle, 1.0)
    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Crop the image (removing specified pixels from each border) after rotation
    remove_border_final = False
    if remove_border_final:
        cropped_rotated_image = corrected_image[border_size:-border_size, border_size:-border_size]
    else:
        cropped_rotated_image = corrected_image

    # Save the corrected image
    out_file = img_file + "_rotated.jpg"
    cv2.imwrite(out_file, cropped_rotated_image)

    return out_file


def test_fastfood():
    assert os.path.isfile(align_image("tests/fastfood.jpg"))
    # can't find box for receipt
    assert align_image("tests/receipt.jpg") is None
    assert os.path.isfile(align_image("tests/rotate-ex2.png"))

    assert os.path.isfile(correct_rotation("tests/fastfood.jpg"))
    assert os.path.isfile(correct_rotation("tests/receipt.jpg"))
    assert os.path.isfile(correct_rotation("tests/rotate-ex2.png"))
