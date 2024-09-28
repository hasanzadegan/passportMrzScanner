from flask import Flask, request, jsonify, send_from_directory
import io
import os
import uuid
import cv2
import numpy as np
import base64
from cropper import process_image
from utils import perform_ocr, resize_image, convert_to_grayscale, apply_processing, process_contours
import json

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    line_number = request.form.get('line_number', type=int, default=1)  # Default to 1 if not provided

    if image_file:
        image_bytes = image_file.read()

        # Convert image_bytes to an OpenCV image
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        height, width = image.shape[:2]

        if width < 1200:
            image = resize_image(image, new_width=1200)

        height, width = image.shape[:2]


        if line_number == 0:
            cropped_img = image
        else:
            cropped_img = process_image(image, line_number)

        eroded_image = apply_processing(cropped_img)
        final_cropped_image = process_contours(eroded_image, cropped_img, 600, 800)


        # Convert final_cropped_image to base64
        _, buffer = cv2.imencode('.jpeg', final_cropped_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Perform OCR
        ocr_result = perform_ocr(final_cropped_image)

        # Ensure ocr_result is a valid JSON
        try:
            ocr_result_json = json.loads(ocr_result)
        except json.JSONDecodeError:
            return jsonify({'error': 'Failed to decode OCR result to JSON'}), 500

        # Include base64 image in the response
        response = {
            'mrz': ocr_result_json,
            'base64': img_base64
        }

        return jsonify(response)

    return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
