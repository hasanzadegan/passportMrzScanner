import cv2
import uuid
import subprocess
import os

def getMRZ(image):
    file_name = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(file_name, image)

    try:
        result = subprocess.check_output(
            ['tesseract', file_name, 'stdout', '--psm', '6', '-l', 'mrz'],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        result = None
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)

    return result

