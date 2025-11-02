import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path
import time

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = 'models'  # Directory where your models are saved
VALIDATION_DIR = 'validation'  # For accuracy testing
IMAGE_SIZE = (28, 28)
MIN_CONFIDENCE = 0.70

# ============================================
# LOAD EXISTING MODELS
# ============================================
def load_existing_models():
    """Load pre-trained models from MODEL_DIR"""
    print("="*60)
    print("LOADING EXISTING MODELS")
    print("="*60)
    
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory '{MODEL_DIR}' not found!")
        return None
    
    # Find all .h5 model files
    model_files = list(Path(MODEL_DIR).glob('*.h5'))
    
    if not model_files:
        print(f"ERROR: No .h5 model files found in '{MODEL_DIR}'")
        return None
    
    print(f"\nFound {len(model_files)} model files:")
    for mf in model_files:
        print(f"  - {mf.name}")
    
    # Load models
    models = []
    for model_path in model_files:
        try:
            model = tf.keras.models.load_model(str(model_path))
            models.append(model)
            print(f"✓ Loaded: {model_path.name}")
        except Exception as e:
            print(f"✗ Error loading {model_path.name}: {e}")
    
    if not models:
        print("\nERROR: Could not load any models!")
        return None
    
    print(f"\n✓ Successfully loaded {len(models)} models")
    print("="*60)
    return models

# ============================================
# TEST ALL VALIDATION FILES
# ============================================
def test_all_validation_files(models):
    """Loop through all validation files and show recognition results"""
    print("\n" + "="*60)
    print("TESTING ALL VALIDATION FILES")
    print("="*60)
    
    if not os.path.exists(VALIDATION_DIR):
        print(f"ERROR: Validation directory '{VALIDATION_DIR}' not found!")
        return
    
    # Collect all image files with their true labels
    test_files = []
    
    for digit in range(10):
        digit_dir = os.path.join(VALIDATION_DIR, str(digit))
        
        if not os.path.exists(digit_dir):
            print(f"Warning: Directory '{digit_dir}' not found, skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(digit_dir).glob(ext))
            image_files.extend(Path(digit_dir).glob(ext.upper()))
        
        for img_path in image_files:
            test_files.append((str(img_path), digit))
        
        print(f"Found {len(image_files)} images for digit {digit}")
    
    if not test_files:
        print("ERROR: No test files found!")
        return
    
    print(f"\n✓ Total test files: {len(test_files)}")
    print("="*60)
    
    # Statistics
    total_correct = 0
    total_wrong = 0
    total_low_confidence = 0
    per_digit_stats = {i: {'correct': 0, 'wrong': 0, 'low_conf': 0, 'total': 0} for i in range(10)}
    
    # Results list for detailed output
    results_list = []
    
    print("\nProcessing images...")
    start_time = time.time()
    
    # Process each file
    for idx, (img_path, true_label) in enumerate(test_files):
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Preprocess
        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        img_input = np.expand_dims(img_input, axis=-1)
        
        # Ensemble prediction
        all_preds = []
        for model in models:
            pred = model.predict(img_input, verbose=0)[0]
            all_preds.append(pred)
        
        avg_pred = np.mean(all_preds, axis=0)
        predicted_label = np.argmax(avg_pred)
        confidence = avg_pred[predicted_label]
        
        # Update statistics
        per_digit_stats[true_label]['total'] += 1
        
        if confidence >= MIN_CONFIDENCE:
            if predicted_label == true_label:
                total_correct += 1
                per_digit_stats[true_label]['correct'] += 1
                status = "✓ CORRECT"
            else:
                total_wrong += 1
                per_digit_stats[true_label]['wrong'] += 1
                status = "✗ WRONG"
        else:
            total_low_confidence += 1
            per_digit_stats[true_label]['low_conf'] += 1
            status = "⚠ LOW CONFIDENCE"
        
        # Store result
        filename = os.path.basename(img_path)
        results_list.append({
            'file': filename,
            'true': true_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'status': status
        })
        
        # Show progress every 50 files
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_files)} files...")
    
    total_time = time.time() - start_time
    
    # ============================================
    # DISPLAY DETAILED RESULTS
    # ============================================
    print("\n" + "="*60)
    print("DETAILED RECOGNITION RESULTS")
    print("="*60)
    print(f"{'File Name':<30} {'True':<6} {'Pred':<6} {'Conf':<8} {'Status':<20}")
    print("-" * 90)
    
    for result in results_list:
        print(f"{result['file']:<30} {result['true']:<6} {result['predicted']:<6} {result['confidence']:<8.3f} {result['status']:<20}")
    
    # ============================================
    # SUMMARY STATISTICS
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_tested = len(test_files)
    total_high_conf = total_correct + total_wrong
    
    print(f"\nTotal files tested: {total_tested}")
    print(f"High confidence predictions: {total_high_conf} ({total_high_conf/total_tested*100:.1f}%)")
    print(f"Low confidence predictions: {total_low_confidence} ({total_low_confidence/total_tested*100:.1f}%)")
    
    if total_high_conf > 0:
        accuracy = total_correct / total_high_conf * 100
        print(f"\n✓ Correct predictions: {total_correct}")
        print(f"✗ Wrong predictions: {total_wrong}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
    
    print(f"\nTime taken: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/total_tested:.4f} seconds")
    
    # ============================================
    # PER-DIGIT STATISTICS
    # ============================================
    print("\n" + "="*60)
    print("PER-DIGIT STATISTICS")
    print("="*60)
    print(f"{'Digit':<8}{'Total':<10}{'Correct':<12}{'Wrong':<10}{'Low Conf':<12}{'Accuracy':<12}")
    print("-" * 70)
    
    for digit in range(10):
        stats = per_digit_stats[digit]
        total = stats['total']
        correct = stats['correct']
        wrong = stats['wrong']
        low_conf = stats['low_conf']
        
        if total > 0:
            high_conf_total = correct + wrong
            if high_conf_total > 0:
                acc = correct / high_conf_total * 100
                print(f"{digit:<8}{total:<10}{correct:<12}{wrong:<10}{low_conf:<12}{acc:.2f}%")
            else:
                print(f"{digit:<8}{total:<10}{correct:<12}{wrong:<10}{low_conf:<12}N/A")
    
    # ============================================
    # ERRORS ANALYSIS (Show wrong predictions)
    # ============================================
    wrong_predictions = [r for r in results_list if r['status'] == "✗ WRONG"]
    
    if wrong_predictions:
        print("\n" + "="*60)
        print("WRONG PREDICTIONS ANALYSIS")
        print("="*60)
        print(f"Total wrong predictions: {len(wrong_predictions)}")
        print(f"\n{'File Name':<30} {'True':<8} {'Predicted':<10} {'Confidence':<12}")
        print("-" * 70)
        
        for result in wrong_predictions[:20]:  # Show first 20 errors
            print(f"{result['file']:<30} {result['true']:<8} {result['predicted']:<10} {result['confidence']:<12.3f}")
        
        if len(wrong_predictions) > 20:
            print(f"\n... and {len(wrong_predictions) - 20} more wrong predictions")
    
    # ============================================
    # LOW CONFIDENCE ANALYSIS
    # ============================================
    low_conf_predictions = [r for r in results_list if r['status'] == "⚠ LOW CONFIDENCE"]
    
    if low_conf_predictions:
        print("\n" + "="*60)
        print("LOW CONFIDENCE PREDICTIONS ANALYSIS")
        print("="*60)
        print(f"Total low confidence predictions: {len(low_conf_predictions)}")
        print(f"\n{'File Name':<30} {'True':<8} {'Predicted':<10} {'Confidence':<12}")
        print("-" * 70)
        
        for result in low_conf_predictions[:10]:  # Show first 10
            print(f"{result['file']:<30} {result['true']:<8} {result['predicted']:<10} {result['confidence']:<12.3f}")
        
        if len(low_conf_predictions) > 10:
            print(f"\n... and {len(low_conf_predictions) - 10} more low confidence predictions")
    
    print("\n" + "="*60)

# ============================================
# RECOGNIZE SINGLE IMAGE
# ============================================
def recognize_image(image_path, models, show_window=True):
    """Recognize digits in an image using loaded models"""
    print("\n" + "="*60)
    print("SINGLE IMAGE RECOGNITION")
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
    low_confidence_count = 0
    
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
        
        # Draw results
        if confidence >= MIN_CONFIDENCE:
            results.append((x, predicted_digit, confidence))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{predicted_digit} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            low_confidence_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            text = f"{predicted_digit}? ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Sort results left to right
    results.sort(key=lambda r: r[0])
    
    # Print results
    recognition_time = time.time() - start_time
    
    print(f"\nHigh confidence digits: {len(results)}")
    print(f"Low confidence digits: {low_confidence_count}")
    
    if results:
        print("\nRecognized digits (left to right):")
        for i, (x_pos, digit, conf) in enumerate(results):
            print(f"  {i+1}. Digit: {digit}, Confidence: {conf:.3f}, Position: x={x_pos}")
        
        sequence = ''.join([str(r[1]) for r in results])
        print(f"\n>>> Sequence: {sequence}")
    
    print(f"\nTime taken: {recognition_time:.4f} seconds")
    print("="*60)
    
    # Display image
    if show_window:
        cv2.imshow('Recognition Result', frame)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return frame, results

# ============================================
# MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DIGIT RECOGNITION - USING EXISTING MODELS")
    print("="*60)
    
    # Step 1: Load existing models
    models = load_existing_models()
    
    if models is None:
        print("\nERROR: Cannot proceed without models!")
        print(f"Please ensure .h5 model files exist in '{MODEL_DIR}' directory")
        exit()
    
    # Step 2: Main menu
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Test all validation files (loop through all)")
        print("2. Test single image")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            # Test all validation files
            test_all_validation_files(models)
            
        elif choice == '2':
            # Test single image
            image_path = input("\nEnter image path: ").strip()
            image_path = image_path.strip('"').strip("'")
            
            if image_path:
                recognize_image(image_path, models, show_window=True)
            
        elif choice == '3':
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")
    
    print("\n" + "="*60)
    print("PROGRAM FINISHED")
    print("="*60)