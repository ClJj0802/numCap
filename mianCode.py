import tensorflow as tf #pip install tensorflow
import cv2 #pip install opencv-python
import numpy as np 
import os
import time
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
TRAIN_DIR = 'train'
TEST_DIR = 'test'
VALIDATION_DIR = 'validation'
MODEL_DIR = 'models'
IMAGE_SIZE = (28, 28)
MIN_CONFIDENCE = 0.70
# ============================================
# STEP 1: LOAD CUSTOM DATASET
# ============================================
def load_custom_dataset(base_dir):
    """
    Load images from directory structure:
    base_dir/0/*.jpg
    base_dir/1/*.jpg
    ...
    base_dir/9/*.jpg
    """
    print(f"\nLoading dataset from: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory '{base_dir}' not found!")
        return None, None
    
    images = []
    labels = []
    
    for digit in range(10):
        digit_dir = os.path.join(base_dir, str(digit))
        
        if not os.path.exists(digit_dir):
            print(f"  Warning: Directory '{digit_dir}' not found, skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(digit_dir).glob(ext))
            image_files.extend(Path(digit_dir).glob(ext.upper()))
        
        print(f"  Digit {digit}: Found {len(image_files)} images")
        
        for img_path in image_files:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"    Warning: Could not load {img_path}")
                continue
            
            # Resize to 28x28
            img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Normalize
            img_normalized = img_resized / 255.0
            
            images.append(img_normalized)
            labels.append(digit)
    
    if len(images) == 0:
        print(f"ERROR: No images loaded from {base_dir}")
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"  Total loaded: {len(images)} images")
    print(f"  Shape: {images.shape}")
    
    return images, labels

# ============================================
# STEP 2: TRAIN MODELS WITH CUSTOM DATA
# ============================================
def train_custom_models():
    """Train 3 different models using custom dataset"""
    print("="*60)
    print("TRAINING MODELS WITH CUSTOM DATASET")
    print("="*60)
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load datasets
    print("\n[1/3] Loading training data...")
    train_images, train_labels = load_custom_dataset(TRAIN_DIR)
    
    print("\n[2/3] Loading test data...")
    test_images, test_labels = load_custom_dataset(TEST_DIR)
    
    print("\n[3/3] Loading validation data...")
    val_images, val_labels = load_custom_dataset(VALIDATION_DIR)
    
    if train_images is None or test_images is None or val_images is None:
        print("\nERROR: Could not load all datasets!")
        print("Please check your directory structure:")
        print("  train/0/, train/1/, ..., train/9/")
        print("  test/0/, test/1/, ..., test/9/")
        print("  validation/0/, validation/1/, ..., validation/9/")
        return False
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training samples:   {len(train_images)}")
    print(f"Test samples:       {len(test_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Model configurations
    model_configs = [
        {
            'name': 'custom_model_1.h5',
            'description': 'Model 1: 128 neurons, ReLU, Adam',
            'layers': [128],
            'activation': 'relu',
            'optimizer': 'adam',
            'epochs': 10
        },
        {
            'name': 'custom_model_2.h5',
            'description': 'Model 2: 256 neurons, ReLU, RMSprop',
            'layers': [256],
            'activation': 'relu',
            'optimizer': 'rmsprop',
            'epochs': 10
        },
        {
            'name': 'custom_model_3.h5',
            'description': 'Model 3: 128 neurons, Tanh, Adam',
            'layers': [128],
            'activation': 'tanh',
            'optimizer': 'adam',
            'epochs': 8
        }
    ]
    
    # Train each model
    for i, config in enumerate(model_configs):
        print("\n" + "="*60)
        print(f"TRAINING MODEL {i+1}/3")
        print("="*60)
        print(config['description'])
        
        model_path = os.path.join(MODEL_DIR, config['name'])
        
        if os.path.exists(model_path):
            print(f"\n{config['name']} already exists.")
            user_input = input("Retrain? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Skipping training...")
                continue
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=IMAGE_SIZE),
            tf.keras.layers.Dense(config['layers'][0], activation=config['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=config['optimizer'],
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nTraining for {config['epochs']} epochs...")
        
        # Train
        history = model.fit(
            train_images, train_labels,
            epochs=config['epochs'],
            validation_data=(test_images, test_labels),
            verbose=1
        )
        
        # Save model
        model.save(model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        # Evaluate on test set
        print("\n--- Test Set Evaluation ---")
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Evaluate on validation set (final check)
        print("\n--- Validation Set Evaluation (Final Check) ---")
        val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=0)
        print(f"Validation Loss:     {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Show per-digit accuracy on validation set
        print("\nPer-digit accuracy on validation set:")
        val_predictions = model.predict(val_images, verbose=0)
        val_pred_labels = np.argmax(val_predictions, axis=1)
        
        for digit in range(10):
            digit_mask = val_labels == digit
            if np.sum(digit_mask) > 0:
                digit_acc = np.mean(val_pred_labels[digit_mask] == digit)
                digit_count = np.sum(digit_mask)
                print(f"  Digit {digit}: {digit_acc:.4f} ({digit_acc*100:.2f}%) - {digit_count} samples")
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    return True

# ============================================
# STEP 3: LOAD MODELS
# ============================================
def load_models():
    """Load all trained models"""
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    models = []
    model_files = ['custom_model_1.h5', 'custom_model_2.h5', 'custom_model_3.h5']
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            print(f"✓ Loaded: {model_file}")
        except Exception as e:
            print(f"✗ Error loading {model_file}: {e}")
    
    if not models:
        print("\nNo models found!")
        return None
    
    print(f"\n✓ Successfully loaded {len(models)} models")
    return models

# ============================================
# STEP 4: EVALUATE ENSEMBLE ON VALIDATION SET
# ============================================
def evaluate_ensemble(models):
    """Evaluate ensemble model on validation set"""
    print("\n" + "="*60)
    print("ENSEMBLE MODEL VALIDATION")
    print("="*60)
    
    # Load validation data
    val_images, val_labels = load_custom_dataset(VALIDATION_DIR)
    
    if val_images is None:
        print("Could not load validation data!")
        return
    
    print(f"\nEvaluating on {len(val_images)} validation samples...")
    
    correct = 0
    predictions_per_digit = {i: {'correct': 0, 'total': 0} for i in range(10)}
    
    for i, (img, true_label) in enumerate(zip(val_images, val_labels)):
        # Prepare image
        img_input = np.expand_dims(img, axis=0)
        img_input = np.expand_dims(img_input, axis=-1)
        
        # Ensemble prediction
        all_preds = []
        for model in models:
            pred = model.predict(img_input, verbose=0)[0]
            all_preds.append(pred)
        
        avg_pred = np.mean(all_preds, axis=0)
        predicted_label = np.argmax(avg_pred)
        confidence = avg_pred[predicted_label]
        
        # Only count high confidence predictions
        if confidence >= MIN_CONFIDENCE:
            predictions_per_digit[true_label]['total'] += 1
            if predicted_label == true_label:
                correct += 1
                predictions_per_digit[true_label]['correct'] += 1
    
    # Calculate accuracy
    total_predicted = sum([predictions_per_digit[i]['total'] for i in range(10)])
    overall_accuracy = correct / total_predicted if total_predicted > 0 else 0
    
    print("\n" + "="*60)
    print("ENSEMBLE VALIDATION RESULTS")
    print("="*60)
    print(f"Minimum confidence threshold: {MIN_CONFIDENCE}")
    print(f"Total validation samples: {len(val_images)}")
    print(f"Samples with confidence >= {MIN_CONFIDENCE}: {total_predicted}")
    print(f"Correct predictions: {correct}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    print("\nPer-digit accuracy (high confidence only):")
    for digit in range(10):
        total = predictions_per_digit[digit]['total']
        correct_digit = predictions_per_digit[digit]['correct']
        if total > 0:
            acc = correct_digit / total
            print(f"  Digit {digit}: {acc:.4f} ({acc*100:.2f}%) - {correct_digit}/{total} correct")
        else:
            print(f"  Digit {digit}: No high-confidence predictions")
    
    print("="*60)

# ============================================
# STEP 5: RECOGNIZE DIGITS IN IMAGE
# ============================================
def recognize_image(image_path, models):
    """Recognize digits in a new image"""
    print("\n" + "="*60)
    print("IMAGE RECOGNITION")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image '{image_path}' not found!")
        return None
    
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("ERROR: Could not read image!")
        return None
    
    print(f"Image size: {frame.shape[1]} x {frame.shape[0]} pixels")
    
    start_time = time.time()
    
    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    results = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if not (100 < area < 10000 and 0.2 < aspect_ratio < 2.0):
            continue
        
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        digit_roi = gray[y1:y2, x1:x2]
        
        if digit_roi.size == 0:
            continue
        
        processed = cv2.resize(digit_roi, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        processed = np.expand_dims(processed, axis=-1)
        
        # Ensemble prediction
        all_preds = []
        for model in models:
            pred = model.predict(processed, verbose=0)[0]
            all_preds.append(pred)
        
        avg_pred = np.mean(all_preds, axis=0)
        predicted_digit = np.argmax(avg_pred)
        confidence = avg_pred[predicted_digit]
        
        if confidence >= MIN_CONFIDENCE:
            results.append((x, predicted_digit, confidence))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{predicted_digit} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    results.sort(key=lambda r: r[0])
    
    print(f"\nRecognized {len(results)} digits:")
    if results:
        for i, (x_pos, digit, conf) in enumerate(results):
            print(f"  {i+1}. Digit: {digit}, Confidence: {conf:.3f}")
        sequence = ''.join([str(r[1]) for r in results])
        print(f"\nSequence: {sequence}")
    
    print(f"Time: {time.time() - start_time:.4f} seconds")
    
    return frame, results

# ============================================
# MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUSTOM DIGIT RECOGNITION SYSTEM")
    print("="*60)
    
    # Check if datasets exist
    if not os.path.exists(TRAIN_DIR):
        print(f"\nERROR: Training directory '{TRAIN_DIR}' not found!")
        print("Please create the following structure:")
        print("  train/0/, train/1/, ..., train/9/")
        print("  test/0/, test/1/, ..., test/9/")
        print("  validation/0/, validation/1/, ..., validation/9/")
        exit()
    
    # Step 1: Train models
    print("\n1. Training models with custom dataset...")
    success = train_custom_models()
    
    if not success:
        print("Training failed. Exiting.")
        exit()
    
    # Step 2: Load models
    print("\n2. Loading trained models...")
    models = load_models()
    
    if models is None:
        print("Could not load models. Exiting.")
        exit()
    
    # Step 3: Validate ensemble
    print("\n3. Validating ensemble model...")
    evaluate_ensemble(models)
    
    # Step 4: Test on custom image (optional)
    test_image = input("\n4. Enter image path to test (or press Enter to skip): ").strip()
    
    if test_image and os.path.exists(test_image):
        result = recognize_image(test_image, models)
        if result is not None:
            frame, digits = result
            cv2.imshow('Recognition Result', frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("PROGRAM FINISHED")
    print("="*60)