import os
import cv2
import numpy as np

def add_margin_to_box(box, margin):
    """Add a margin around a given box."""
    center = np.mean(box, axis=0)
    vectors = box - center
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    new_box = box + norm_vectors * margin

    new_box[0, 1] -= margin
    new_box[1, 1] -= margin
    new_box[2, 1] += margin
    new_box[3, 1] += margin

    return new_box.astype(int)

def crop_to_box(image, box):
    """Crop the image to the bounding box defined by the box coordinates."""
    x_min = int(min(box[:, 0]))
    x_max = int(max(box[:, 0]))
    y_min = int(min(box[:, 1]))
    y_max = int(max(box[:, 1]))

    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1] - 1)
    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0] - 1)

    return image[y_min:y_max, x_min:x_max]

def process_image(image):
    """Process the input image to detect and crop regions of interest."""
    output_file = 'output/crop_output.jpg'  # Adjust as needed
    closed_image_file = 'output/closed_image.jpg'  # File for closed_image
    valid_contours_file = 'output/validcontours.jpg'  # File for valid contours

    if image is None:
        print("Input image not found or could not be loaded.")
        return None

    # Resize the image if its width is less than the target
    target_width = 1200
    height, width = image.shape[:2]
    if width < target_width:
        ratio = target_width / float(width)
        new_dimensions = (target_width, int(height * ratio))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    # Convert to grayscale and apply morphological transformations
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 25))
    blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    horizontal_filtered_image = cv2.morphologyEx(blackhat_image, cv2.MORPH_OPEN, horizontal_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    closed_image = cv2.morphologyEx(horizontal_filtered_image, cv2.MORPH_CLOSE, close_kernel)
    cv2.imwrite(closed_image_file, closed_image)

    # Thresholding to get binary image
    _, binary_image = cv2.threshold(closed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    valid_contours_low = []

    # Filter valid contours based on aspect ratio and dimensions
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=int)

        width, height = rect[1]
        if width < height:
            width, height = height, width

        aspect_ratio = width / float(height) if height != 0 else 0
        image_width = image.shape[1]

        if aspect_ratio > 20 and width > 0.2 * image_width and height > 18:
            valid_contours.append(contour)

        if aspect_ratio > 5 and width > 0.2 * image_width and height > 10:
            valid_contours_low.append(contour)

    '''
    print("valid_contours len:", len(valid_contours))
    print("valid_contours_low len:", len(valid_contours_low))
    '''

    if len(valid_contours) < 2:
        valid_contours = valid_contours_low


    # Draw valid contours on the original image
    # cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)  # Green color for contours
    # cv2.imwrite(valid_contours_file, image)  # Save the image with contours

    first_two_contours = valid_contours[:2] if len(valid_contours) >= 2 else valid_contours

    if len(first_two_contours) == 2:
        combined_box = np.vstack([cv2.boxPoints(cv2.minAreaRect(contour)) for contour in first_two_contours])
        merged_rect = cv2.minAreaRect(combined_box)
        merged_box = cv2.boxPoints(merged_rect)
        merged_box = np.array(merged_box, dtype=int)

        margin_box = add_margin_to_box(merged_box, 20)
        center = (int(merged_rect[0][0]), int(merged_rect[0][1]))
        angle = merged_rect[2]

        # Normalize the angle for rotation
        if angle < -45:
            angle += 90

        # Correct the angle based on box dimensions
        width, height = merged_rect[1]
        if width < height:
            angle -= 90  # Adjust if height is greater than width

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_margin_box = cv2.transform(np.array([margin_box]), rotation_matrix)[0]
        cropped_rotated_image = crop_to_box(rotated_image, rotated_margin_box)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, cropped_rotated_image)

        return cropped_rotated_image

    print("Not enough valid contours found.")
    return None
