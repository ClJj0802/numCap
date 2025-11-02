import tensorflow as tf
import cv2
import numpy as np
import os

# --- GPU 检查 ---
print("--- 检查 GPU 可用性 ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"检测到 GPU: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("未检测到 GPU，将在 CPU 上运行。")
print("-------------------------")

# --- 数据加载 (假设你的数据已按类别组织在文件夹中) ---
image_height = 32
image_width = 32
batch_size = 32

# 请将 'your_dataset_root' 替换为你的数据集根目录
# 例如：'my_digit_dataset/' 且内部包含 train/0, train/1, ... 等文件夹
data_root = r'C:\Users\emmanuel thian\Desktop\num_capture' # <-- 确保这个路径指向你的数据集根目录

# Use a temporary variable to get class_names before caching/prefetching
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_root, 'train'),
    labels='inferred',
    label_mode='int',
    image_size=(image_height, image_width),
    interpolation='area',
    batch_size=batch_size,
    shuffle=True
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_root, 'validation'),
    labels='inferred',
    label_mode='int',
    image_size=(image_height, image_width),
    interpolation='area',
    batch_size=batch_size,
    shuffle=False
)

raw_test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_root, 'test'),
    labels='inferred',
    label_mode='int',
    image_size=(image_height, image_width),
    interpolation='area',
    batch_size=batch_size,
    shuffle=False
)

# Get class_names BEFORE applying .cache() and .prefetch()
num_classes = len(raw_train_ds.class_names)
class_names = raw_train_ds.class_names # You might want to store class names for later use
print(f"检测到的类别数量: {num_classes} (对应数字 0-{num_classes-1})")
print(f"类别名称: {class_names}")


# 归一化函数
def normalize_img(image, label):
    # Ensure 3 channels for RGB, even if input images are grayscale (converted from PNG/JPG)
    # The Conv2D layer input_shape=(image_height, image_width, 3) expects 3 channels.
    # If your original images are indeed grayscale, you might need to adjust the Conv2D input_shape to 1.
    # For now, let's assume they are loaded as 3-channel (which image_dataset_from_directory often does by default).
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = raw_train_ds.map(normalize_img)
val_ds = raw_val_ds.map(normalize_img)
test_ds = raw_test_ds.map(normalize_img)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# --- 构建 CNN 模型 ---
model = tf.keras.Sequential([
    # Make sure the input shape matches your image data (e.g., (32, 32, 3) for RGB or (32, 32, 1) for grayscale)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)), # Assuming RGB input
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- 训练模型 ---
epochs = 10 # 根据需要调整
print(f"\n--- 开始训练模型 (共 {epochs} 轮) ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 评估模型 ---
print("\n--- 在测试集上评估模型 ---")
loss, accuracy = model.evaluate(test_ds)
print(f"测试集准确率: {accuracy*100:.2f}%")

# --- 保存训练好的模型 ---
model_save_path = 'my_custom_digit_recognizer_model.h5'
model.save(model_save_path)
print(f"模型已保存到: {model_save_path}")

print("\n自定义模型训练和保存完成。")