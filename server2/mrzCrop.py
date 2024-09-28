import os
import cv2
import numpy as np

def add_margin_to_box(box, margin):
    """Add a margin around a given box."""
    center = np.mean(box, axis=0)
    vectors = box - center
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    new_box = box + norm_vectors * margin

    # Adjust the top and bottom edges of the box to add vertical margin
    new_box[0, 1] -= margin
    new_box[1, 1] -= margin
    new_box[2, 1] += margin
    new_box[3, 1] += margin

    return new_box.astype(int)

def crop_to_box(image, box):
    """Crop the image to the bounding box defined by the box coordinates."""
    x_min = max(0, int(min(box[:, 0])))
    x_max = min(int(max(box[:, 0])), image.shape[1] - 1)
    y_min = max(0, int(min(box[:, 1])))
    y_max = min(int(max(box[:, 1])), image.shape[0] - 1)

    return image[y_min:y_max, x_min:x_max]

def clean_image(image, block_size=11, constant=5):
    """Convert the image to grayscale and apply adaptive thresholding."""
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 3:
        if image.shape[2] == 3 or image.shape[2] == 4:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    elif len(image.shape) == 2:
        gray_image = image  # Image is already grayscale
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant)

    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/clean_output.jpg', cleaned_image)

    return cleaned_image


def process_image(image, saveFiles=False):
    """Process the input image to detect and crop regions of interest."""
    if image is None:
        raise ValueError("Input image not found or could not be loaded.")

    # Ensure output directories exist
    os.makedirs('output', exist_ok=True)

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

    # Thresholding to get a binary image
    _, binary_image = cv2.threshold(closed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mrz_contours = []
    mrz_contours_low = []
    barcode_contours = []

    # Filter valid contours based on aspect ratio and dimensions
    image_width = image.shape[1]
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = np.array(cv2.boxPoints(rect), dtype=int)

        width, height = rect[1]
        if width < height:
            width, height = height, width

        aspect_ratio = width / float(height) if height != 0 else 0

        if aspect_ratio > 25 and width > 0.25 * image_width and height > 18:
            mrz_contours.append(contour)

        if aspect_ratio > 8 and width > 0.2 * image_width and height > 10:
            mrz_contours_low.append(contour)

        if aspect_ratio < 8 and width > 0.1 * image_width and 40 < height < 80:
            barcode_contours.append(contour)

    # Fallback to lower threshold if fewer than 2 MRZ contours are found
    if len(mrz_contours) < 2:
        mrz_contours = mrz_contours_low

    if len(mrz_contours) >= 2:
        # Combine the first two MRZ contours into one bounding box
        combined_box = np.vstack([cv2.boxPoints(cv2.minAreaRect(contour)) for contour in mrz_contours[:2]])
        merged_rect = cv2.minAreaRect(combined_box)
        merged_box = np.array(cv2.boxPoints(merged_rect), dtype=int)

        # Add margin and rotate the image
        margin_box = add_margin_to_box(merged_box, 20)
        center = tuple(map(int, merged_rect[0]))
        angle = merged_rect[2]

        # Normalize the angle for rotation
        if angle < -45:
            angle += 90

        # Correct the angle based on box dimensions
        width, height = merged_rect[1]
        if width < height:
            angle -= 90

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image and crop to the adjusted bounding box
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_margin_box = cv2.transform(np.array([margin_box]), rotation_matrix)[0]
        cropped_rotated_image = crop_to_box(rotated_image, rotated_margin_box)

        # Save intermediate images if requested
        if saveFiles:
            cv2.imwrite('output/1.gray.jpg', gray_image)
            cv2.imwrite('output/2.blackhat.jpg', blackhat_image)
            cv2.imwrite('output/3.horizontal_filtered.jpg', horizontal_filtered_image)
            cv2.imwrite('output/4.closed_image.jpg', closed_image)
            cv2.imwrite('output/5.binary_image.jpg', binary_image)
            cv2.imwrite('output/crop_output.jpg', cropped_rotated_image)

        return cropped_rotated_image

    print("Not enough valid contours found.")
    return None
