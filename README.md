# MRZ Detection and OCR API

This project is a Flask-based web service for detecting Machine Readable Zones (MRZ) in images and extracting text using Tesseract OCR. The service is designed to process images, detect MRZ regions, and provide OCR results in a structured JSON format.

## Features

- **Image Processing:** Enhances image quality through resizing, grayscale conversion, and morphological operations.
- **MRZ Detection:** Identifies and extracts MRZ regions from images, specifically tailored for passport and ID document scanning.
- **OCR Integration:** Leverages Tesseract OCR with a custom-trained model (`mrz.traineddata`) for precise MRZ text extraction.
- **API Endpoint:** Offers a RESTful API endpoint for uploading images and receiving processed results.

## Requirements

- **Python 3.x**: Ensure you have Python 3.x installed on your system.
- **Flask**: A lightweight WSGI web application framework.
- **OpenCV**: A library for computer vision tasks.
- **NumPy**: A library for numerical operations in Python.
- **Tesseract OCR**: An open-source OCR engine used for text extraction.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
