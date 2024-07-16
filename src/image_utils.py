import os

import numpy as np
from scipy.stats import mode

from utils import have_cv2, have_pillow
from enums import images_num_max_dict


def largest_contour(contours):
    """ Find the largest contour in the list. """
    import cv2
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    return largest_contour


def is_contour_acceptable(contour, image, size_threshold=0.1, aspect_ratio_range=(0.5, 2), rotation_threshold=30):
    import cv2
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


def file_to_cv2(img_file):
    import cv2
    image = cv2.imread(img_file)
    assert os.path.isfile(img_file), '%s not found' % img_file
    if image is None:
        # e.g. small BW gif gridnumbers.gif
        from PIL import Image
        import numpy as np
        pil_image = Image.open(img_file).convert('RGB')
        pil_image_file = img_file + '.pil.png'
        pil_image.save(pil_image_file)
        image = cv2.imread(pil_image_file)
        # open_cv_image = np.array(pil_image, dtype=np.unit8)
        ## Convert RGB to BGR
        # image = open_cv_image[:, :, ::-1].copy()

    # Check if image is loaded
    if image is None:
        raise ValueError("Error: Image for %s not made." % img_file)
    return image


def align_image(img_file):
    import cv2
    from imutils.perspective import four_point_transform
    try:
        # Load the image
        # img_file = '/home/jon/Downloads/fastfood.jpg'
        # img_file = "/home/jon/Documents/reciept.jpg"
        image = file_to_cv2(img_file)
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
                return img_file
        else:
            print("No acceptable contours found.")
            return img_file
    except Exception as e:
        print("Error in align_image:", e, flush=True)
        return img_file


def correct_rotation(img_file, border_size=50):
    import cv2
    # Function to rotate the image to the correct orientation
    # Load the image
    image = file_to_cv2(img_file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect points that form a line using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return img_file

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


def pad_resize_image_file(img_file, relaxed_resize=False):
    import cv2

    image = file_to_cv2(img_file)
    if relaxed_resize:
        postfix = "_resized.png"
        image = resize_image(image, return_none_if_no_change=True, max_dimension=2048)
    else:
        postfix = "_pad_resized.png"
        image = pad_resize_image(image, return_none_if_no_change=True)
    if image is None:
        new_file = img_file
    else:
        new_file = img_file + postfix
        cv2.imwrite(new_file, image)

    return new_file


def resize_image(image, return_none_if_no_change=True, max_dimension=2048):
    import cv2
    height, width = image.shape[:2]

    # Calculate the scaling factor
    if max(height, width) > max_dimension:
        if height > width:
            scale_factor = max_dimension / height
        else:
            scale_factor = max_dimension / width

        # Compute new dimensions
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))

        # Resize the image
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    else:
        # No resizing needed if the image is already within the desired dimensions
        if return_none_if_no_change:
            return None
        resized_image = image
    return resized_image


def pad_resize_image(image, return_none_if_no_change=False, max_dimension=1024):
    import cv2

    L = max_dimension
    H = max_dimension

    # Load the image
    Li, Hi = image.shape[1], image.shape[0]

    if Li == L and Hi == H:
        if return_none_if_no_change:
            return None
        else:
            return image

    # Calculate the aspect ratio
    aspect_ratio_original = Li / Hi
    aspect_ratio_final = L / H

    # Check the original size and determine the processing needed
    if Li < L and Hi < H:
        # Padding
        padding_x = (L - Li) // 2
        padding_y = (H - Hi) // 2
        image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
    elif Li > L and Hi > H:
        # Resizing
        if aspect_ratio_original < aspect_ratio_final:
            # The image is taller than the target aspect ratio
            new_height = H
            new_width = int(H * aspect_ratio_original)
        else:
            # The image is wider than the target aspect ratio
            new_width = L
            new_height = int(L / aspect_ratio_original)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        # Intermediate case, resize without cropping
        if aspect_ratio_original < aspect_ratio_final:
            # The image is taller than the target aspect ratio
            new_height = H
            new_width = int(H * aspect_ratio_original)
        else:
            # The image is wider than the target aspect ratio
            new_width = L
            new_height = int(L / aspect_ratio_original)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padding_x = (L - new_width) // 2
        padding_y = (H - new_height) // 2
        image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])

    # debug, to see effect of pad-resize
    # import cv2
    # cv2.imwrite('new1.png', image)

    return image


def fix_image_file(file, do_align=False, do_rotate=False, do_pad=False, relaxed_resize=False):
    # always try to fix rotation/alignment since OCR better etc. in that case
    if have_cv2:
        if do_align:
            aligned_image = align_image(file)
            if aligned_image is not None and os.path.isfile(aligned_image):
                file = aligned_image
        if do_rotate:
            derotated_image = correct_rotation(file)
            if derotated_image is not None and os.path.isfile(derotated_image):
                file = derotated_image
        if do_pad or relaxed_resize:
            file = pad_resize_image_file(file, relaxed_resize=relaxed_resize)
    return file


def get_image_types():
    if have_pillow:
        from PIL import Image
        exts = Image.registered_extensions()
        image_types0 = {ex for ex, f in exts.items() if f in Image.OPEN}
        image_types0 = sorted(image_types0)
        image_types0 = [x[1:] if x.startswith('.') else x for x in image_types0]
    else:
        image_types0 = []
    return image_types0


def get_image_file(image_file, image_control, document_choice, base_model=None, images_num_max=None,
                   image_resolution=None, image_format=None,
                   convert=False,
                   str_bytes=True):
    if image_control is not None:
        img_file = image_control
    elif image_file is not None:
        img_file = image_file
    else:
        image_types = get_image_types()
        img_file = [x for x in document_choice if
                    any(x.endswith('.' + y) for y in image_types)] if document_choice else []

    if not isinstance(img_file, list):
        img_file = [img_file]
    if isinstance(img_file, list) and not img_file:
        img_file = [None]

    final_img_files = []
    for img_file1 in img_file:
        if convert:
            if img_file1 and os.path.isfile(img_file1):
                from vision.utils_vision import img_to_base64
                img_file1 = img_to_base64(img_file1, str_bytes=str_bytes, resolution=image_resolution,
                                          output_format=image_format)
            elif isinstance(img_file1, str):
                # assume already bytes
                img_file1 = img_file1
            else:
                img_file1 = None
        final_img_files.append(img_file1)
    final_img_files = [x for x in final_img_files if x]
    if base_model and images_num_max == -1:
        images_num_max = images_num_max_dict.get(base_model, 1)
    if base_model and images_num_max is None:
        images_num_max = images_num_max_dict.get(base_model, 1) or 1
    if images_num_max is None:
        images_num_max = len(final_img_files)
    if images_num_max <= -1:
        images_num_max = -images_num_max - 1
    final_img_files = final_img_files[:images_num_max]
    return final_img_files
