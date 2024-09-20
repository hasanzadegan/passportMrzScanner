import cv2
import numpy as np
import os

def add_margin_to_box(box, margin):
    center = np.mean(box, axis=0)
    vectors = box - center
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    new_box = box + norm_vectors * margin

    new_box[0, 1] -= margin  # Top left
    new_box[1, 1] -= margin  # Top right
    new_box[2, 1] += margin  # Bottom right
    new_box[3, 1] += margin  # Bottom left

    return new_box.astype(int)

def crop_to_box(image, box):
    x_min = int(min(box[:, 0]))
    x_max = int(max(box[:, 0]))
    y_min = int(min(box[:, 1]))
    y_max = int(max(box[:, 1]))

    # Ensure the coordinates are within image bounds
    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1] - 1)
    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0] - 1)

    return image[y_min:y_max, x_min:x_max]

def process_image(input_file, output_file):
    print("Step 1: Reading the input image.")
    image = cv2.imread(input_file)

    if image is None:
        print(f"Error: Image {input_file} not found or could not be loaded.")
        return None

    print("Step 2: Resizing the image if necessary.")
    target_width = 1200
    height, width = image.shape[:2]
    if width < target_width:
        ratio = target_width / float(width)
        new_dimensions = (target_width, int(height * ratio))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    print("Step 3: Converting the image to grayscale.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Step 4: Applying blackhat morphology.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 25))
    blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    print("Step 5: Filtering horizontal components.")
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    horizontal_filtered_image = cv2.morphologyEx(blackhat_image, cv2.MORPH_OPEN, horizontal_kernel)

    print("Step 6: Closing gaps in the filtered image.")
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    closed_image = cv2.morphologyEx(horizontal_filtered_image, cv2.MORPH_CLOSE, close_kernel)

    print("Step 7: Binarizing the image.")
    _, binary_image = cv2.threshold(closed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    print("Step 8: Filtering valid contours.")
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=int)

        width, height = rect[1]
        if width < height:
            width, height = height, width

        aspect_ratio = width / float(height) if height != 0 else 0
        image_width = image.shape[1]

        if aspect_ratio > 25 and width > 0.2 * image_width and height > 18:
            valid_contours.append(contour)

    first_two_contours = valid_contours[:2] if len(valid_contours) >= 2 else valid_contours

    if len(first_two_contours) == 2:
        print("Step 9: Combining the boxes of the first two contours.")
        combined_box = np.vstack([cv2.boxPoints(cv2.minAreaRect(contour)) for contour in first_two_contours])
        merged_rect = cv2.minAreaRect(combined_box)
        merged_box = cv2.boxPoints(merged_rect)
        merged_box = np.array(merged_box, dtype=int)

        print("Step 10: Adding margin to the combined box.")
        margin_box = add_margin_to_box(merged_box, 20)
        center = (int(merged_rect[0][0]), int(merged_rect[0][1]))

        width, height = merged_rect[1]
        angle = merged_rect[2]

        print("Step 11: Rotating the image.")
        if height < width:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)

        if abs(angle) > 90:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle - 180, 1.0)

        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_margin_box = cv2.transform(np.array([margin_box]), rotation_matrix)[0]

        print("Step 12: Cropping the rotated image.")
        cropped_rotated_image = crop_to_box(rotated_image, rotated_margin_box)

        cv2.imwrite(output_file, cropped_rotated_image)
        print(f"Step 13: Cropped rotated image saved to {output_file}.")
        return cropped_rotated_image

    print("Error: Not enough valid contours found.")
    return None

# Specify input and output file paths
input_file = '../img/p15.jpeg'  # Change this to your specific file
output_file = 'output/crop_p15.jpg'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Process the image
process_image(input_file, output_file)
