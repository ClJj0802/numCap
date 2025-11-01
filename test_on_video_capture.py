import tensorflow as tf
import cv2
import numpy as np
import time


#### 如果泡不到
#pip uninstall opencv-python opencv-python-headless -y
#pip install opencv-python


# --- 載入已訓練的模型 ---
model = tf.keras.models.load_model('my_custom_digit_recognizer_model.h5')
image_height = 32
image_width = 32
class_names = [str(i) for i in range(10)]  # 根據你的分類數字 0–9

# --- 開啟攝影機 ---
cap = cv2.VideoCapture(1)  # 改為 0/1 取決於你的攝影機
if not cap.isOpened():
    print("無法打開攝影機")
    exit()

print("攝影機已開啟，按 'q' 結束")

# --- FPS 控制參數 ---
target_display_fps = 4
target_frame_duration = 1.0 / target_display_fps
last_processed_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法從攝影機讀取畫面")
        break

    current_time = time.time()

    if (current_time - last_processed_time) >= target_frame_duration:
        # --- 擷取 ROI (假設數字出現在中央區域，可根據需要調整) ---
        h, w, _ = frame.shape
        roi_size = 200
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (image_width, image_height))
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)  # [1, H, W, 3]

        # --- 預測 ---
        predictions = model.predict(roi_input)
        predicted_label = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # --- 顯示結果 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"Prediction: {predicted_label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # --- 顯示畫面 ---
        cv2.imshow("Digit Recognition", frame)
        last_processed_time = current_time

    # --- 鍵盤控制 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- 清理資源 ---
cap.release()
cv2.destroyAllWindows()
print("結束辨識")
