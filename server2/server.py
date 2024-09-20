from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from mrzCrop import process_image
from mrzDetector import perform_ocr

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

    if image_file:
        image_bytes = image_file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode the image'}), 400

        # Process the image using the existing function
        cropped_image = process_image(image)

        if cropped_image is None:
            return jsonify({'error': 'Image processing failed'}), 500

        # Perform OCR on the cropped image
        mrz_json = perform_ocr(cropped_image)

        return jsonify(mrz_json)

    return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
