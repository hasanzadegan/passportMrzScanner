import cv2
import numpy as np
import os
import pytesseract
import json
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
import json
import uuid
import re



def rotate_image_bound(image, angle, center=None, scale=1.0):
    """Rotate the image around the specified center by the given angle."""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_angle(angle):
    """Correct the angle to be in the range [0, 180) degrees."""
    if angle < -45:
        angle += 90
    elif angle >= 45:
        angle -= 90
    return angle

def resize_image(image, new_width=1200):
    """Resize image to the specified width while maintaining aspect ratio."""
    height, width = image.shape[:2]
    aspect_ratio = new_width / float(width)
    new_height = int(height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def convert_to_grayscale(image):
    """Convert image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
    return grayscale_image

def apply_gaussian_blur(image, kernel_size=(13, 13)):
    """Apply Gaussian blur to the image."""
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def apply_blackhat(image, kernel):
    """Apply blackhat operation to highlight dark regions on a light background."""
    blackhat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return blackhat_image

def apply_scharr(image):
    """Apply Scharr operator to detect edges."""
    image = np.float32(image)
    scharr_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)

    scharr = cv2.magnitude(scharr_x, scharr_y)
    scharr = np.uint8(np.clip(scharr, 0, 255))
    return scharr

def apply_closing(image, kernel):
    """Apply closing operation to close small gaps in the image."""
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def apply_closing2(image, kernel_size):
    """Apply morphological closing operation with a specified kernel size."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def create_mask(image, threshold=128, min_area=400):
    """Create a binary mask from the image using a threshold and remove small white areas."""
    _, mask_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))  # Adjust kernel size as needed
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask_image)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def apply_erosion(image, kernel_size=(3, 3), iterations=1):
    """Apply erosion to reduce noise in the image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    cv2.imwrite(f"./uploads/eroded_{uuid.uuid4()}.jpeg", eroded_image)

    return eroded_image

def rotate_image(image, angle, center=None):
    """Rotate image by the specified angle."""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def draw_rectangle_around_contour(image, contour):
    """Draw a rectangle around the contour on a copy of the image."""
    if contour is None:
        print("No contour provided.")
        return image

    if len(contour) == 0:
        print("Empty contour.")
        return image

    # Get the minimum area rectangle for the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # Updated from np.int0 to np.int32

    # Create a copy of the image to draw on
    result_image = image.copy()

    # Draw the rectangle on the image
    if len(result_image.shape) == 2:  # Check if the image is grayscale
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)  # Convert to color if necessary

    cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2)  # Green rectangle

    return result_image

def pad_line1(line, target_length=44):
    """Ensure the first '<' is at the second character, then pad or trim the line to the target length for line 1."""
    lt_index = line.find('<')

    # maybe not correct for other country
    if lt_index != -1:
        if lt_index > 1:
            # Move the first '<' to the second position
            line = line[lt_index-1:]  # Keep characters starting from one before the '<'
        elif lt_index == 0:
            # If the first '<' is at position 0, adjust the line
            line = '<' + line[1:]
    else:
        # If there's no '<', add it to the second position
        line = line[:1] + '<' + line[1:]

    # Ensure the length of the line is exactly target_length
    if len(line) < target_length:
        # Pad the line with '<' if it's shorter than the target length
        return line.ljust(target_length, '<')
    else:
        # Trim the line from the end if it's longer than the target length
        return line[:target_length]


def pad_line2(line, target_length=44):
    line = line.replace(" ", "")
    if len(line) > target_length:
        line = line[:target_length]
    elif len(line) < target_length:
        line = line.ljust(target_length, '<')
    return line


def numberToLetter(c):
    return "OLZEASGTBP"[int(c)] if c.isdigit() else c

def correct_passport_number(passport_number):
    # Replace any non-alphanumeric characters with an empty string
    passport_number = ''.join(c for c in passport_number if c.isalnum())

    # Replace the first character if it's a digit
    if passport_number and passport_number[0].isdigit():
        passport_number = numberToLetter(passport_number[0]) + passport_number[1:]

    return passport_number


def convert_dob(dob_str):
    """Convert a date of birth string in YYMMDD format to yy-mm-dd format."""
    try:
        if len(dob_str) != 6:
            raise ValueError("Date of birth string must be 6 characters long.")

        year = int(dob_str[0:2])
        month = int(dob_str[2:4])
        day = int(dob_str[4:6])

        current_year = datetime.now().year
        current_century = current_year // 100 * 100
        if year > current_year % 100:
            year += (current_century - 100)
        else:
            year += current_century

        dob = datetime(year, month, day)
        return dob.strftime('%y-%m-%d')

    except Exception as e:
        return f"Error: {str(e)}"



def apply_processing(image):
    # resized_image = resize_image(image, new_width=1200)
    resized_image = image
    grayscale_image = convert_to_grayscale(resized_image)
    blurred_image = apply_gaussian_blur(grayscale_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (resized_image.shape[1] // 30, 14))
    blackhat_image = apply_blackhat(blurred_image, kernel)
    scharr_image = apply_scharr(blackhat_image)
    closed_image = apply_closing(scharr_image, kernel)
    mask_image = create_mask(closed_image)
    closing_kernel_size = (resized_image.shape[1] // 30, 35)
    closed_again = apply_closing2(mask_image, closing_kernel_size)
    eroded_image = apply_erosion(closed_again)
    return eroded_image


def process_contours(eroded_image, image, min_y_position, min_width):
    """Process contours to return the final cropped image."""
    # Find contours
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        filtered_contours = [
            cnt for cnt in contours
            if cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] > min_y_position and
               cv2.boundingRect(cnt)[2] >= min_width
        ]

        if filtered_contours:
            bottom_contour = max(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3])

            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(bottom_contour)
            margin = 25
            x = max(x - margin, 0)
            y = max(y - margin, 0)

            # Ensure the cropped area does not exceed image bounds
            w = min(w + margin, image.shape[1] - x)
            h = min(h + margin, image.shape[0] - y)

            # Crop the image
            cropped_image = image[y:y+h, x:x+w]

            # Convert cropped image to grayscale
            grayscale_image = convert_to_grayscale(cropped_image)

            # Get rotated bounding box
            rect = cv2.minAreaRect(bottom_contour)
            angle = correct_angle(rect[2])
            rotated_image = rotate_image_bound(grayscale_image, angle, center=(w // 2, h // 2))

            # Final crop based on height
            h = rotated_image.shape[0]
            if h > 400:
                final_cropped_image = rotated_image[h - 400:, :]
            else:
                final_cropped_image = rotated_image

            cv2.imwrite(f"./uploads/final_cropped_{uuid.uuid4()}.jpeg", final_cropped_image)


            return final_cropped_image
        else:
            print("No suitable contours found.")
            return None
    else:
        print("No contours found.")
        return None


def perform_ocr(image):
    """Perform OCR using Tesseract with MRZ language and page segmentation mode 6."""
    try:
        text = pytesseract.image_to_string(image, lang='mrz', config='--psm 6')
        text = text.strip()

        if text:
            text = text.replace('\n', ' ')
            lines = text.split(' ')

            filtered_lines = [line.strip() for line in lines if len(line.strip()) > 35]
            filtered_lines = filtered_lines[-2:]  # Use the last two lines

            if len(filtered_lines) >= 2:
                line1 = pad_line1(filtered_lines[0], 44)
                line2 = pad_line2(filtered_lines[1], 44)
                print(text)
                print("line1",line1)
                print("line2",line2)

                if len(line1) == 44 and len(line2) == 44:
                    passport_number = correct_passport_number(line2[0:9].strip())
                    date_of_birth = convert_dob(line2[13:19].strip())
                    date_of_expire = convert_dob(line2[21:27].strip())

                    mrz_fields = {
                        'document_type': line1[0],
                        'country_code': line1[2:5],
                        'last_name': line1[5:44].split('<')[0].strip(),
                        'middle_name': line1[5:44].split('<')[1].strip(),
                        'first_name': line1[5:44].split('<<')[1].split('<')[0].strip() if '<' in line1[5:44] else '',
                        'passport_number': passport_number,
                        'nationality': line2[10:13].strip(),
                        'date_of_birth': date_of_birth,
                        'gender': line2[20].strip(),
                        'expiration_date': date_of_expire,
                        'personal_number': line2[28:42].strip(),
                        'check_digit': line2[43].strip(),
                    }

                    return json.dumps(mrz_fields, indent=4)
                else:
                    return json.dumps({
                        "error": "MRZ lines do not have the expected length of 44 characters.",
                        "detected_text": text
                    }, indent=4)
            else:
                return json.dumps({
                    "error": "Not enough lines detected or lines are too short.",
                    "detected_text": text
                }, indent=4)
        else:
            return json.dumps({
                "error": "No text detected.",
                "detected_text": text
            }, indent=4)
    except Exception as e:
        return json.dumps({
            "error": f"Error during OCR: {str(e)}",
            "detected_text": text if 'text' in locals() else ''
        }, indent=4)
