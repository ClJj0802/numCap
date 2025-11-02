import tensorflow as tf
import cv2
import numpy as np
import os
import time # Import the time module for timing
# --- 1. Load your pre-trained MNIST models ---
# Make sure you have 'mnist_model_1.h5', 'mnist_model_2.h5', 'mnist_model_3.h5'
# in the same directory as this script.
# The training block below will create them if they don't exist.

# Example training and saving multiple MNIST models (run once for each model):
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# --- Training and saving Model 1 ---
if not os.path.exists('mnist_model_1.h5'):
    print("Training Model 1...")
    model_for_training_1 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_for_training_1.compile(optimizer='adam',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=['accuracy'])
    model_for_training_1.fit(train_images, train_labels, epochs=5, verbose=0)
    model_for_training_1.save('mnist_model_1.h5')
    print("MNIST Model 1 trained and saved as mnist_model_1.h5")

# --- Training and saving Model 2 ---
if not os.path.exists('mnist_model_2.h5'):
    print("Training Model 2...")
    model_for_training_2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_for_training_2.compile(optimizer='rmsprop',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=['accuracy'])
    model_for_training_2.fit(train_images, train_labels, epochs=5, verbose=0)
    model_for_training_2.save('mnist_model_2.h5')
    print("MNIST Model 2 trained and saved as mnist_model_2.h5")

# --- Training and saving Model 3 ---
if not os.path.exists('mnist_model_3.h5'):
    print("Training Model 3...")
    model_for_training_3 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_for_training_3.compile(optimizer='adam',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=['accuracy'])
    model_for_training_3.fit(train_images, train_labels, epochs=3, verbose=0)
    model_for_training_3.save('mnist_model_3.h5')
    print("MNIST Model 3 trained and saved as mnist_model_3.h5")


# List of model paths to load
model_paths = ['mnist_model_1.h5', 'mnist_model_2.h5', 'mnist_model_3.h5']
loaded_models = []

for path in model_paths:
    try:
        current_model = tf.keras.models.load_model(path)
        loaded_models.append(current_model)
        print(f"Successfully loaded model from {path}.")
    except Exception as e:
        print(f"Could not load model from {path}: {e}")
        print(f"Please ensure '{path}' exists and is a valid TensorFlow Keras model.")

if not loaded_models:
    print("Error: No models were loaded. Exiting.")
    exit()

print(f"Loaded {len(loaded_models)} models for ensembling.")

# --- Define the minimum confidence threshold ---
MIN_CONFIDENCE_THRESHOLD = 0.70 # Adjust this value as needed (e.g., 0.8, 0.9)

# --- Define the image path ---
image_path = '1_10num.jpg' # Make sure this image file is in the same directory

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file '{image_path}' was not found.")
    print("Please make sure '1_10num.jpg' is in the same directory as this script.")
    exit()

# --- Load the image ---
print(f"Loading image: {image_path}")
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}. Check file path and permissions.")
    exit()

print("Image loaded successfully. Starting number recognition.")

# --- Start timing the recognition process ---
start_time = time.time()

# --- Process the loaded image ---
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
_, thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    area = cv2.contourArea(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    min_area = 100
    max_area = 10000
    min_aspect_ratio = 0.5 # Adjusted for potential variations in digits
    max_aspect_ratio = 1.8 # Adjusted for potential variations in digits

    # Filter contours based on area and aspect ratio
    if area > min_area and area < max_area and \
       min_aspect_ratio < aspect_ratio < max_aspect_ratio:

        padding = 10
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
        digit_roi = gray_frame[y1:y2, x1:x2]

        if digit_roi.shape[0] == 0 or digit_roi.shape[1] == 0:
            continue # Skip empty ROIs

        processed_digit = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        processed_digit = processed_digit / 255.0
        input_image = np.expand_dims(processed_digit, axis=0)
        input_image = np.expand_dims(input_image, axis=-1) # Add channel dimension for TensorFlow

        # --- Perform Ensemble Prediction ---
        all_predictions = []
        for model in loaded_models:
            model_pred = model.predict(input_image, verbose=0)[0]
            all_predictions.append(model_pred)

        # Average the probabilities (Soft Voting)
        averaged_predictions = np.mean(all_predictions, axis=0)
        predicted_label = np.argmax(averaged_predictions)
        confidence = averaged_predictions[predicted_label] # Confidence from averaged probabilities

        # --- Draw Results on Original Frame ---
        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            # Draw bounding box (green box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display prediction result and confidence
            text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

# --- End timing the recognition process ---
end_time = time.time()
recognition_time = end_time - start_time

# --- Display Processed Frame ---
cv2.imshow('Image Recognition Result', frame)

print(f"\nRecognition completed for '{image_path}'.")
print(f"Time taken for recognition: {recognition_time:.4f} seconds.")

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Program exited.")