import cv2
import pytesseract
import base64

def pad_line(line, length):
    """Pad the line to the specified length with spaces."""
    return line.ljust(length)

def correct_passport_number(passport_number):
    passport_number = passport_number.strip().replace("<", "")
    return passport_number

def convert_dob(dob):
    """Convert date of birth from YYMMDD to DD/MM/YYYY format."""
    return f"{dob[:2]}/{dob[2:4]}/{dob[4:]}"

def encode_image_to_base64(image):
    """Encode the image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def perform_ocr(image):
    """Perform OCR on the provided image and extract MRZ fields."""
    try:
        text = pytesseract.image_to_string(image, lang='mrz', config='--psm 6').strip()
        text = text.replace(' ', '')
        # print("text",text)

        if text:
            # Replace new lines with spaces and split into words
            lines = text.replace('\n', ' ').split(' ')
            # Filter lines longer than 35 characters

            if len(lines) >= 2:
                line1 = pad_line(lines[0], 44)
                line2 = pad_line(lines[1], 44)

                '''
                print("line1:", line1)
                print("line1 length:", len(line1))
                print("line2:", line2)
                print("line2 length:", len(line2))
                '''

                if len(line1) == 44 and len(line2) == 44:
                    passport_number = correct_passport_number(line2[0:9].strip())
                    date_of_birth = convert_dob(line2[13:19].strip())
                    date_of_expire = convert_dob(line2[21:27].strip())

                    mrz_fields = {
                        'document_type': line1[0],
                        'country_code': line1[2:5],
                        'last_name': line1[5:44].split('<')[0].strip(),
                        'middle_name': line1[5:44].split('<')[1].strip() if '<' in line1[5:44] else '',
                        'first_name': line1[5:44].split('<<')[1].split('<')[0].strip() if '<<' in line1[5:44] else '',
                        'passport_number': passport_number,
                        'nationality': line2[10:13].strip(),
                        'date_of_birth': date_of_birth,
                        'gender': line2[20].strip(),
                        'expiration_date': date_of_expire,
                        'personal_number': line2[28:42].strip(),
                        'check_digit': line2[43].strip(),
                    }

                    # Encode the image to base64
                    base64_image = encode_image_to_base64(image)

                    return {
                        'base64': base64_image,
                        'mrz': mrz_fields,
                        'line1': line1,
                        'line2': line2
                    }  # Return the base64 image and MRZ fields
                else:
                    return {
                        "error": "MRZ lines do not have the expected length of 44 characters.",
                        "detected_text": text
                    }
            else:
                return {
                    "error": "Not enough lines detected or lines are too short.",
                    "detected_text": text
                }
        else:
            return {
                "error": "No text detected.",
                "detected_text": text
            }
    except Exception as e:
        return {
            "error": f"Error during OCR: {str(e)}",
            "detected_text": text if 'text' in locals() else ''
        }
