import cv2
import numpy as np
import os

input_image_path = '../img/p507.jpeg'  # Specify the single image
output_directory = 'output'
os.makedirs(output_directory, exist_ok=True)

# Clear the output directory
for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Process the specified image
image = cv2.imread(input_image_path)

if image is None:
    print(f"Image {input_image_path} not found or could not be loaded.")
else:
    print("Image loaded successfully.")

    target_width = 1200
    height, width = image.shape[:2]
    if width < target_width:
        ratio = target_width / float(width)
        new_dimensions = (target_width, int(height * ratio))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box

        # Crop the image to the bounding box
        cropped_image = image[y:y + h, x:x + w]

        # Save cropped image
        cv2.imwrite(os.path.join(output_directory, 'p507_Cropped_Largest_Contour.jpeg'), cropped_image)

    # Save other images
    images_to_save = {
        '1.Gray': gray_image,
        '2.Binary': binary_image,
        'crop_output': image,  # Save the image with all contours and the largest contour box
    }

    # Save all images in one go
    for key, img in images_to_save.items():
        cv2.imwrite(os.path.join(output_directory, f'{key}.jpg'), img)

    print(f"Processed p507.jpeg, saved steps 1 to 4 and cropped the largest contour.")
