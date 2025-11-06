import cv2
import os
import pytesseract
from pathlib import Path

# ================== 配置 ==================
# 设置 Tesseract 路径（Windows 用户必须设置！）
# 下载地址：https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

train_dir = str(Path(__file__).parent.resolve())  # 当前脚本所在目录（即图片目录）
output_root = 'cropped_results'
os.makedirs(output_root, exist_ok=True)

print(f"📁 正在处理目录: {train_dir}")
print(f"📁 输出目录: {output_root}")

total_expected = 0
total_saved = 0

# 遍历所有图片文件
for filename in os.listdir(train_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    if filename == os.path.basename(__file__) or filename == output_root:
        continue

    img_path = os.path.join(train_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像，跳过: {filename}")
        continue

    total_expected += 1

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理：增强对比度 + 二值化（适合模糊/低对比度图像）
    # 先高斯模糊降噪
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 自适应阈值（比 Otsu 更适合不均匀光照）
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    # 使用 Tesseract 获取每个字符的位置和内容
    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
    try:
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"   ❌ Tesseract 处理失败: {filename} | 错误: {e}")
        continue

    digit_count = 0

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text.isdigit():  # 只保留纯数字
            continue

        # 检查置信度是否有效
        conf_str = data['conf'][i]
        if conf_str == '-1':
            continue
        try:
            conf = float(conf_str)
        except ValueError:
            continue

        x = int(data['left'][i])
        y = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])

        # 过滤掉太小或置信度过低的框
        if conf < 50 or w < 5 or h < 5:
            continue

        # 裁剪数字区域
        crop_img = img[y:y+h, x:x+w]

        # 添加 padding（防止裁太紧）
        pad = 3
        y_pad = max(0, y - pad)
        x_pad = max(0, x - pad)
        h_pad = min(img.shape[0] - y_pad, h + 2 * pad)
        w_pad = min(img.shape[1] - x_pad, w + 2 * pad)
        crop_img = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

        # 保存裁剪结果
        digit_count += 1
        output_name = f"{Path(filename).stem}_digit_{digit_count}.jpg"
        save_path = os.path.join(output_root, output_name)
        cv2.imwrite(save_path, crop_img)
        total_saved += 1

        print(f"   🖼️ 裁剪第{digit_count}个数字: '{text}' @ ({x},{y}) | 置信度: {conf:.1f}%")

    if digit_count == 0:
        print(f"   ⚠️ 未检测到任何数字: {filename}")

# ===== 最终报告 =====
print(f"\n✅ 处理完成！")
print(f"   📷 共发现 {total_expected} 张有效图片")
print(f"   💾 成功保存 {total_saved} 个数字裁剪结果到 '{output_root}'")