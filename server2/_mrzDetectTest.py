import os
import cv2
from mrzCrop import process_image,clean_image  # Make sure the function is accessible
from mrzDetector import perform_ocr  # Ensure this function is accessible
def process_images_in_directory(input_directory):
    """Process all images in the specified directory to detect MRZ fields."""

    os.makedirs('./output/result/not_detected', exist_ok=True)
    os.makedirs('./output/result/detected', exist_ok=True)


    total_images = 0
    successful_detections = 0

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            file_path = os.path.join(input_directory, filename)

            # Read the image
            image = cv2.imread(file_path)

            # Crop the image using mrzCrop.py
            cropped_image = process_image(image, 0)

            if cropped_image is not None:
                # Perform OCR on the cropped image using mrzDetector.py
                mrz_json = perform_ocr(cropped_image)

                passport_number = mrz_json.get('mrz', {}).get('passport_number', None)
                document_type = mrz_json.get('mrz', {}).get('document_type', None)

                if passport_number is None or 'error' in mrz_json or document_type != 'P':
                    cropped_image = clean_image(cropped_image, 31, 5)
                    mrz_json = perform_ocr(cropped_image, False)

                passport_number = mrz_json.get('mrz', {}).get('passport_number', None)

                p1 = passport_number[1] if passport_number is not None and len(passport_number) > 1 else 0
                if passport_number is None or not p1.isdigit():
                    cropped_image = clean_image(cropped_image, 11, 5)
                    mrz_json = perform_ocr(cropped_image, False)

                line1 = mrz_json.get('line1', None)
                l1 = line1[0] if line1 is not None and len(line1) > 1 else ' '

                if passport_number and l1 == 'P':
                    successful_detections += 1
                    print(f"--- {filename} ------- {passport_number} --- {line1[0]}")
                    cv2.imwrite(f"output/result/detected/{passport_number}-{filename}.jpg", cropped_image)
                else:
                    print(f"--- {filename} ------- No passport number detected.")
                    cv2.imwrite(f"output/result/not_detected/{passport_number}-{filename}.jpg", cropped_image)


    if total_images > 0 :
        accuracy = (successful_detections / total_images) * 100
        print(f"Detection accuracy: {accuracy:.2f}% from {total_images}")
    else:
        print("No images processed.")

if __name__ == "__main__":
    input_directory = '../img'
    process_images_in_directory(input_directory)

