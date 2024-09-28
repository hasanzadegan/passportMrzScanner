import os
import cv2
from mrzCrop import process_image,clean_image  # Make sure the function is accessible
from mrzDetector import perform_ocr  # Ensure this function is accessible

def process_images_in_directory(input_directory):
    """Process all images in the specified directory to detect MRZ fields."""
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_directory, filename)

            # Read the image
            image = cv2.imread(file_path)

            # Crop the image using mrzCrop.py
            cropped_image = process_image(image)

            if cropped_image is not None:
                # Perform OCR on the cropped image using mrzDetector.py
                result = perform_ocr(cropped_image)

                if 'error' in result:
                    cleaned_image = clean_image(cropped_image)
                    result = perform_ocr(cleaned_image)

                passport_number = result.get('mrz', {}).get('passport_number', None)

                if passport_number is None:
                    cleaned_image = clean_image(cropped_image)
                    result = perform_ocr(cleaned_image)


                passport_number = result.get('mrz', {}).get('passport_number', None)

                cv2.imwrite(f"output/result/{passport_number}-{filename}.jpg", cropped_image)


                if passport_number:
                    print(f"--- {filename} ------- {passport_number}")
                else:
                    print(f"--- {filename} ------- No passport number detected.")

if __name__ == "__main__":
    input_directory = '../img'
    process_images_in_directory(input_directory)
