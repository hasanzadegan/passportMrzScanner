import cv2

# Read the image
input_file = 'output/crop_output.jpg'
output_file = 'output/crop_output1.jpg'

# Load the image
image = cv2.imread(input_file)

if image is not None:
    # Convert to grayscale
    gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary_image2 = cv2.adaptiveThreshold(
        gray_image2,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # Remove noise using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # You can adjust the kernel size
    cleaned_image = cv2.morphologyEx(binary_image2, cv2.MORPH_OPEN, kernel)

    # Save the cleaned binary image
    cv2.imwrite(output_file, cleaned_image)
    print(f"Cleaned binary image saved to {output_file}.")
else:
    print("Error: Image not found or could not be loaded.")
