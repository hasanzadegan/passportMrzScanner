import cv2
import numpy as np
import os

def calculate_line_length(x1, y1, x2, y2):
    """Calculate the length of a line segment."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    """Calculate the angle of the line segment with the horizontal axis."""
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def is_within_angle_range(angle, min_angle=-40.0, max_angle=40.0):
    """Check if the angle is within the specified range."""
    return min_angle <= angle <= max_angle

def group_lines(lines, angle_tolerance=2.0, distance_tolerance=50):
    """Group lines that are close in angle and distance."""
    grouped_lines = []
    visited = [False] * len(lines)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        if visited[i]:
            continue

        angle = calculate_angle(x1, y1, x2, y2)
        if not is_within_angle_range(angle):
            continue

        group = [(x1, y1, x2, y2)]
        visited[i] = True

        for j, (x1_other, y1_other, x2_other, y2_other) in enumerate(lines):
            if visited[j]:
                continue

            angle1 = calculate_angle(x1, y1, x2, y2)
            angle2 = calculate_angle(x1_other, y1_other, x2_other, y2_other)
            if abs(angle1 - angle2) < angle_tolerance:
                distance = abs(np.cross([x2 - x1, y2 - y1], [x1_other - x1, y1_other - y1]) /
                               np.linalg.norm([x2 - x1, y2 - y1]))
                if distance < distance_tolerance:
                    group.append((x1_other, y1_other, x2_other, y2_other))
                    visited[j] = True

        grouped_lines.append(group)

    return grouped_lines

def average_line(group):
    """Calculate the average line from a group of lines."""
    x1_avg = y1_avg = x2_avg = y2_avg = 0
    n = len(group)

    for (x1, y1, x2, y2) in group:
        x1_avg += x1
        y1_avg += y1
        x2_avg += x2
        y2_avg += y2

    return (x1_avg // n, y1_avg // n, x2_avg // n, y2_avg // n)

def extend_line_to_edges(x1, y1, x2, y2, img_width, img_height):
    """Extend a line segment to the edges of the image."""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:  # vertical line
        x1_new = x2_new = x1
        y1_new = 0
        y2_new = img_height
    else:
        slope = dy / dx
        intercept = y1 - slope * x1

        x1_new = 0
        y1_new = int(slope * x1_new + intercept)

        x2_new = img_width
        y2_new = int(slope * x2_new + intercept)

        if y1_new < 0:
            y1_new = 0
            x1_new = int((y1_new - intercept) / slope)
        elif y1_new > img_height:
            y1_new = img_height
            x1_new = int((y1_new - intercept) / slope)

        if y2_new < 0:
            y2_new = 0
            x2_new = int((y2_new - intercept) / slope)
        elif y2_new > img_height:
            y2_new = img_height
            x2_new = int((y2_new - intercept) / slope)

    return (x1_new, y1_new, x2_new, y2_new)

def rotate_image(image, angle):
    """Rotate the image by the specified angle."""
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    return rotated_image

def crop_image_from_line(image, line):
    """Crop the image from the detected line up to the top edge."""
    x1, y1, x2, y2 = line
    if y1 > y2:
        y1, y2 = y2, y1

    # Find the topmost y-coordinate of the red line
    top_y = min(y1, y2)

    # Crop the image from the topmost point of the line to the top edge of the image
    cropped_image = image[0:top_y, :]  # Crop from the top edge to the top of the line

    return cropped_image

def process_image(image, line_number):
    """Process the image to detect lines, rotate, and crop based on the specified line number."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    image_width = image.shape[1]
    image_height = image.shape[0]
    min_line_length = 0.10 * image_width  # 10% of the image width

    if lines is not None:
        filtered_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                line_length = calculate_line_length(x1, y1, x2, y2)
                if line_length > min_line_length:
                    filtered_lines.append((x1, y1, x2, y2))

        # Group lines and calculate the average line for each group
        grouped_lines = group_lines(filtered_lines)

        # Sort lines based on their y-coordinate to find the lowest lines
        all_lines = []
        for group in grouped_lines:
            avg_line = average_line(group)
            all_lines.append(avg_line)

        # Sort lines by their average y-coordinates of the endpoints
        all_lines.sort(key=lambda line: max(line[1], line[3]), reverse=True)

        # Select the requested line
        if 1 <= line_number <= len(all_lines):
            x1, y1, x2, y2 = all_lines[line_number - 1]
        else:
            raise ValueError(f"Line number {line_number} is out of range. Valid range is 1 to {len(all_lines)}.")

        # Draw the selected line on the image (for visualization)
        annotated_img = image.copy()
        extended_line = extend_line_to_edges(x1, y1, x2, y2, image_width, image_height)
        cv2.line(annotated_img, (extended_line[0], extended_line[1]), (extended_line[2], extended_line[3]), (0, 0, 255), 2)  # Drawing in red

        # Rotate and crop the image based on the detected line
        angle = calculate_angle(x1, y1, x2, y2)
        rotated_img = rotate_image(annotated_img, angle)

        # Crop the rotated image based on the detected line
        cropped_img = crop_image_from_line(rotated_img, (x1, y1, x2, y2))

        return cropped_img

    else:
        raise ValueError("No lines were detected in the image.")
