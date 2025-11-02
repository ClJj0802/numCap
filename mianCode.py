import tensorflow as tf
import cv2
import numpy as np
import os
import time

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = 'models'
IMAGE_PATH = '1_10num.jpg'  # Change this to your image file
MIN_CONFIDENCE = 0.70

# ============================================
# STEP 1: TRAIN AND SAVE MODELS
# ============================================
def train_mnist_models():
    """Train 3 different MNIST models"""
    print("="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    print(f"Training samples: {len(train_images)}, Test samples: {len(test_images)}")
    
    # Model 1: Dense 128 + ReLU
    model_path_1 = os.path.join(MODEL_DIR, 'mnist_model_1.h5')
    if not os.path.exists(model_path_1):
        print("\n[1/3] Training Model 1 (128 neurons, ReLU, Adam)...")
        model1 = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model1.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model1.fit(train_images, train_labels, epochs=5, verbose=1)
        model1.save(model_path_1)
        acc = model1.evaluate(test_images, test_labels, verbose=0)[1]
        print(f"Model 1 saved! Test accuracy: {acc:.4f}")
    else:
        print("[1/3] Model 1 already exists")
    
    # Model 2: Dense 256 + ReLU
    model_path_2 = os.path.join(MODEL_DIR, 'mnist_model_2.h5')
    if not os.path.exists(model_path_2):
        print("\n[2/3] Training Model 2 (256 neurons, ReLU, RMSprop)...")
        model2 = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model2.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model2.fit(train_images, train_labels, epochs=5, verbose=1)
        model2.save(model_path_2)
        acc = model2.evaluate(test_images, test_labels, verbose=0)[1]
        print(f"Model 2 saved! Test accuracy: {acc:.4f}")
    else:
        print("[2/3] Model 2 already exists")
    
    # Model 3: Dense 128 + Tanh
    model_path_3 = os.path.join(MODEL_DIR, 'mnist_model_3.h5')
    if not os.path.exists(model_path_3):
        print("\n[3/3] Training Model 3 (128 neurons, Tanh, Adam)...")
        model3 = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model3.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model3.fit(train_images, train_labels, epochs=3, verbose=1)
        model3.save(model_path_3)
        acc = model3.evaluate(test_images, test_labels, verbose=0)[1]
        print(f"Model 3 saved! Test accuracy: {acc:.4f}")
    else:
        print("[3/3] Model 3 already exists")
    
    print("\nAll models ready!\n")

# ============================================
# STEP 2: LOAD MODELS
# ============================================
def load_models():
    """Load all trained models"""
    print("="*50)
    print("LOADING MODELS")
    print("="*50)
    
    models = []
    model_files = ['mnist_model_1.h5', 'mnist_model_2.h5', 'mnist_model_3.h5']
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            print(f"✓ Loaded: {model_file}")
        except Exception as e:
            print(f"✗ Error loading {model_file}: {e}")
    
    if not models:
        print("No models found!")
        return None
    
    print(f"\nSuccessfully loaded {len(models)} models\n")
    return models

# ============================================
# STEP 3: RECOGNIZE DIGITS IN IMAGE
# ============================================
def recognize_digits(image_path, models):
    """Recognize digits in the image"""
    print("="*50)
    print("RECOGNIZING DIGITS")
    print("="*50)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return None
    
    # Load image
    print(f"\nLoading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("ERROR: Could not read the image!")
        return None
    
    print(f"Image size: {frame.shape[1]} x {frame.shape[0]} pixels")
    
    # Start timing
    start_time = time.time()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    # Process each contour
    results = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Filter by area and aspect ratio
        if not (100 < area < 10000 and 0.2 < aspect_ratio < 2.0):
            continue
        
        # Extract digit region with padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        digit_roi = gray[y1:y2, x1:x2]
        
        if digit_roi.size == 0:
            continue
        
        # Preprocess for model
        processed = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        processed = np.expand_dims(processed, axis=-1)
        
        # Ensemble prediction
        all_preds = []
        for model in models:
            pred = model.predict(processed, verbose=0)[0]
            all_preds.append(pred)
        
        # Average predictions
        avg_pred = np.mean(all_preds, axis=0)
        predicted_digit = np.argmax(avg_pred)
        confidence = avg_pred[predicted_digit]
        
        # Only use high confidence predictions
        if confidence >= MIN_CONFIDENCE:
            results.append((x, predicted_digit, confidence))
            
            # Draw on image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{predicted_digit} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Sort results left to right
    results.sort(key=lambda r: r[0])
    
    # Print results
    recognition_time = time.time() - start_time
    
    print(f"\nRecognized {len(results)} digits:")
    if results:
        for i, (x_pos, digit, conf) in enumerate(results):
            print(f"  {i+1}. Digit: {digit}, Confidence: {conf:.3f}, Position: x={x_pos}")
        
        sequence = ''.join([str(r[1]) for r in results])
        print(f"\nSequence (left to right): {sequence}")
    
    print(f"Time taken: {recognition_time:.4f} seconds")
    print("="*50)
    
    return frame, results

# ============================================
# MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("MNIST DIGIT RECOGNITION")
    print("="*50 + "\n")
    
    # Step 1: Train models (if needed)
    train_mnist_models()
    
    # Step 2: Load models
    models = load_models()
    
    if models is None:
        print("Cannot proceed without models. Exiting.")
        exit()
    
    # Step 3: Recognize digits
    result = recognize_digits(IMAGE_PATH, models)
    
    if result is not None:
        frame, digits = result
        
        # Display result
        cv2.imshow('Digit Recognition Result', frame)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\nProgram finished.")