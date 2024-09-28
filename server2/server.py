from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
from mrzCrop import process_image, clean_image
from mrzDetector import perform_ocr

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    if 'base64' not in request.form:
        return jsonify({'error': 'No base64 string provided'}), 400

    base64_string = request.form['base64']
    header, encoded = base64_string.split(',', 1)  # Split header from the actual base64
    image_data = base64.b64decode(encoded)  # Decode the base64 string
    np_array = np.frombuffer(image_data, np.uint8)  # Convert to numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode the image

    if image is None:
        return jsonify({'error': 'Could not decode the base64 image'}), 400

    # Process the image using the existing function
    cropped_image = process_image(image,1)

    if cropped_image is None:
        return jsonify({'error': 'Image processing failed'}), 500

    # Perform OCR on the cropped image
    mrz_json = perform_ocr(cropped_image, False)
    passport_number = mrz_json.get('mrz', {}).get('passport_number', None)

    if passport_number is None or 'error' in mrz_json:
        cropped_image = clean_image(cropped_image,31,5)
        mrz_json = perform_ocr(cropped_image, False)

    passport_number = mrz_json.get('mrz', {}).get('passport_number', None)

    p1 = passport_number[1] if passport_number is not None and len(passport_number) > 1 else 0
    if passport_number is None or not p1.isdigit():
        cropped_image = clean_image(cropped_image,11,5)
        mrz_json = perform_ocr(cropped_image, False)

    # Encode cropped image to base64 for response
    _, buffer = cv2.imencode('.jpg', cropped_image)
    base64_cropped = base64.b64encode(buffer).decode('utf-8')
    mrz_json['base64'] = base64_cropped

    return jsonify(mrz_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
