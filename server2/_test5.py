import cv2
from pyzbar import pyzbar
import numpy as np
import os

# Function to detect and decode barcodes
def detect_and_decode_barcode(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines in the image using HoughLinesP to correct rotation if needed
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # If lines are detected, find the angle and rotate the image to deskew
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        # Calculate the median angle
        median_angle = np.median(angles)

        # Rotate the image by the median angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        # If no lines are detected, use the original image
        rotated_image = image

    # Detect the barcodes in the (potentially rotated) image
    barcodes = pyzbar.decode(rotated_image)

    # Store the largest barcode found
    largest_barcode = None
    largest_size = 0

    # Iterate over all detected barcodes
    for barcode in barcodes:
        # Extract the bounding box of the barcode
        (x, y, w, h) = barcode.rect
        size = w * h  # Calculate the area of the barcode's bounding box

        # Check if this is the largest barcode we've found so far
        if size > largest_size:
            largest_size = size
            largest_barcode = barcode

    return largest_barcode

# Function to process all images in the directory and print the largest barcode in each image
def process_directory(directory):
    # Iterate over all image files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Modify extensions as needed
            image_path = os.path.join(directory, filename)
            print(f"Processing image: {image_path}")

            # Detect the largest barcode in the current image
            barcode = detect_and_decode_barcode(image_path)

            if barcode:
                # Extract the largest barcode data
                barcode_data = barcode.data.decode("utf-8")
                (x, y, w, h) = barcode.rect
                print(f"Largest barcode in {filename}: {barcode_data} (Size: {w * h})")
            else:
                print(f"No barcodes found in {filename}")

# Specify the directory containing the images
input_directory = '../img'

# Process the directory and print the largest barcode in each image
process_directory(input_directory)
