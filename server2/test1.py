import cv2
import numpy as np
import os
import pytesseract

input_directory = '../img'
output_directory = 'output'
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(input_directory):
    input_image_path = os.path.join(input_directory, filename)
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Image {filename} not found or could not be loaded.")
        continue

    target_width = 1200
    height, width = image.shape[:2]
    if width < target_width:
        ratio = target_width / float(width)
        new_dimensions = (target_width, int(height * ratio))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 25))
    blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal_filtered_image = cv2.morphologyEx(blackhat_image, cv2.MORPH_OPEN, horizontal_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    closed_image = cv2.morphologyEx(horizontal_filtered_image, cv2.MORPH_CLOSE, close_kernel)

    _, mask = cv2.threshold(closed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(closed_image, gray_image, mask=mask)

    final_closed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    final_closed_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, final_closed_kernel)

    contours, _ = cv2.findContours(final_closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    image_width = image.shape[1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0.3 * image_width and (h / float(w)) < 0.3 and (h / float(w)) > 0.1:
            filtered_contours.append(contour)

    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)  # Draw green borders

    # Save the results
    cv2.imwrite(os.path.join(output_directory, f'1.Gray_{filename}'), gray_image)
    cv2.imwrite(os.path.join(output_directory, f'2.Blackhat_{filename}'), blackhat_image)
    cv2.imwrite(os.path.join(output_directory, f'3.Horizontal_Filter_{filename}'), horizontal_filtered_image)
    cv2.imwrite(os.path.join(output_directory, f'4.Closed_{filename}'), closed_image)
    cv2.imwrite(os.path.join(output_directory, f'5.Masked_{filename}'), masked_image)
    cv2.imwrite(os.path.join(output_directory, f'6.Final_Closed_{filename}'), final_closed_image)
    cv2.imwrite(os.path.join(output_directory, f'7.Contours_{filename}'), image)  # Ensure this saves the image with contours

    print(f"Processed {filename}, saved Contours image as '7.Contours_{filename}'")

print(f"Processed images have been saved in the output directory: {output_directory}")
