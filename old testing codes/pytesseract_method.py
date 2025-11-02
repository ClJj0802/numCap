import cv2
import pytesseract
import numpy as np
import os
import re

# --- IMPORTANT: Keep this line as it resolved your PATH issue ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the image path
image_path = '1_10num.jpg'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file '{image_path}' was not found.")
    print("Please make sure '1_10num.jpg' is in the same directory as this script.")
    exit()

# Load the image
print(f"Loading image: {image_path}")
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}. Check file path and permissions.")
    exit()

print("Image loaded successfully. Starting number recognition using Tesseract.")

# --- Image Preprocessing for OCR ---
# Convert to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise (helps with OCR for slightly blurry or noisy images)
# kernel_size can be (3,3) or (5,5)
blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

# Apply Otsu's thresholding (often good for text)
# This converts the image to pure black and white, which Tesseract prefers.
# Using cv2.THRESH_BINARY_INV can be helpful if numbers are dark on a light background.
_, thresh_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --- DISPLAY THE PREPROCESSED IMAGE (FOR DEBUGGING) ---
# This is crucial to see what Tesseract is actually "seeing"
cv2.imshow('Preprocessed Image for Tesseract', thresh_frame)
cv2.waitKey(1) # Display for a short moment, allowing the main window to open later


# --- Tesseract OCR Configuration ---
# Experiment with different PSMs.
# PSM 6: Assume a single uniform block of text. (Good general choice)
# PSM 7: Treat the image as a single text line. (Could work if numbers are on one row)
# PSM 8: Treat the image as a single word. (If each number is truly isolated)
# PSM 10: Treat the image as a single character. (Very strict, but can be powerful for isolated chars)

# Start with a less strict PSM and ensure character whitelist.
# We'll try PSM 6 first, then PSM 8 if needed.
tesseract_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
# tesseract_config = r'--psm 8 -c tessedit_char_whitelist=0123456789' # Uncomment to try this

# Use the preprocessed (thresholded) image for OCR
data = pytesseract.image_to_data(thresh_frame, config=tesseract_config, output_type=pytesseract.Output.DICT)

n_boxes = len(data['text'])
min_confidence = 60 # Lower the minimum confidence threshold for initial debugging.
                    # We can raise it later once we start seeing detections.

print("\n--- Tesseract Raw Detections (for debugging) ---")
found_any_number = False

for i in range(n_boxes):
    conf = float(data['conf'][i])
    text = data['text'][i]
    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

    # Print all raw detections, even low confidence, to see what Tesseract is finding
    # Filter out empty strings or very low confidence early for cleaner debug output
    if text.strip() and conf > -1: # conf > -1 means it's a valid detection, even if low conf
        print(f"Raw: '{text}' (Conf: {conf:.0f}%) BBox: ({x},{y},{w},{h})")

    # --- Post-processing and Filtering ---
    cleaned_text = re.sub(r'[^0-9]', '', text).strip()

    # Check if the confidence is high enough AND if it's a valid digit
    if conf > min_confidence and cleaned_text:
        try:
            number = int(cleaned_text)

            # Filter for numbers typically found in a 1-10 chart
            # This is your application-specific logic.
            if 1 <= number <= 10:
                found_any_number = True
                # Draw bounding box and text on the ORIGINAL frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                display_text = f"{number} ({conf:.0f}%)"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                print(f"--> DETECTED & ACCEPTED: {number} at ({x},{y},{w},{h}) with confidence {conf:.2f}%")

        except ValueError:
            # This block will be hit if cleaned_text is empty or not a valid integer after cleaning
            pass

print("\n--- Recognition Summary ---")
if found_any_number:
    print("Numbers were detected and accepted based on confidence and range.")
else:
    print("No numbers were detected with sufficient confidence or within the specified range (1-10).")
    print("Please review the 'Raw Detections' above and the 'Preprocessed Image'.")

# Display the final processed frame
cv2.imshow('Final Recognition Result (Tesseract)', frame)

print("\nRecognition completed.")

# Wait for a key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Program exited.")