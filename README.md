# Passport MRZ Scanner

This project provides a Flask-based web service for detecting Machine Readable Zones (MRZ) in passport and ID card images and extracting text using Tesseract OCR. It processes images to detect MRZ regions and uses a custom-trained Tesseract model for accurate text extraction.

## Features

- **Image Processing:** Resizes, converts images to grayscale, and applies morphological operations for enhanced quality.
- **MRZ Detection:** Detects MRZ regions in passport and ID card images.
- **OCR Integration:** Utilizes Tesseract OCR with a custom-trained model (`mrz.traineddata`) for precise MRZ text extraction.
- **API Endpoint:** Provides a RESTful API for uploading images and receiving processed results.

## Requirements

- **Python 3.x**: [Download Python](https://www.python.org/downloads/)
- **Flask**: A lightweight WSGI web application framework. [Flask Documentation](https://flask.palletsprojects.com/)
- **OpenCV**: A library for computer vision tasks. [OpenCV Documentation](https://docs.opencv.org/)
- **NumPy**: A library for numerical operations in Python. [NumPy Documentation](https://numpy.org/doc/)
- **Tesseract OCR**: An open-source OCR engine used for text extraction. [Tesseract GitHub Releases](https://github.com/tesseract-ocr/tesseract)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hasanzadegan/passportMrzScanner.git
   cd passportMrzScanner
