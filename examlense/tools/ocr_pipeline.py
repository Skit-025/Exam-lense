import cv2 as cv
import numpy as np
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# =========================
# LOAD MODEL (only once)
# =========================
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# =========================
# LINE SEGMENTATION (BEST)
# =========================
def detect_and_crop_lines(img_path, output_folder):
    img = cv.imread(img_path)
    if img is None:
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    inverted = cv.bitwise_not(binary)

    # Remove notebook lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
    removed_lines = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel)
    clean = cv.subtract(inverted, removed_lines)

    # Row projection
    row_sum = np.sum(clean, axis=1)
    row_sum = cv.GaussianBlur(row_sum.reshape(-1, 1), (7, 1), 0).flatten()

    threshold = np.percentile(row_sum, 65)
    line_indices = np.where(row_sum > threshold)[0]

    lines = []
    if len(line_indices) > 0:
        gap = int(img.shape[0] * 0.012)
        start = line_indices[0]

        for i in range(1, len(line_indices)):
            if line_indices[i] - line_indices[i - 1] > gap:
                lines.append((start, line_indices[i - 1]))
                start = line_indices[i]

        lines.append((start, line_indices[-1]))

    # Filter lines
    heights = [end - start for (start, end) in lines]
    final_lines = []

    if len(heights) > 0:
        avg_height = np.mean(heights)

        for (start, end) in lines:
            h = end - start
            if avg_height * 0.5 < h < avg_height * 1.8:
                final_lines.append((start, end))

    # Crop
    os.makedirs(output_folder, exist_ok=True)

    paths = []
    for i, (start_y, end_y) in enumerate(final_lines):
        padding = int((end_y - start_y) * 0.3)

        start_y = max(0, start_y - padding)
        end_y = min(img.shape[0], end_y + padding)

        crop = img[start_y:end_y, :]

        if crop.shape[0] > 15:
            path = os.path.join(output_folder, f"line_{i}.jpg")
            cv.imwrite(path, crop)
            paths.append(path)

    return paths


# =========================
# QUALITY CHECK
# =========================
def is_good_line(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    non_zero = np.count_nonzero(gray < 200)
    ratio = non_zero / (img.shape[0] * img.shape[1])

    if ratio < 0.02:
        return False

    h = img.shape[0]
    if h < 15 or h > 150:
        return False

    return True


# =========================
# OCR (TrOCR)
# =========================
def recognize_line(img_path):
    image = Image.open(img_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# =========================
# CONFIDENCE FILTER
# =========================
def is_confident(text):
    if len(text.strip()) < 3:
        return False
    if text.count("?") > 2:
        return False
    return True


# =========================
# MAIN PIPELINE
# =========================
def run_ocr(image_path):
    crop_folder = "temp_lines"

    # Step 1: Segment
    line_paths = detect_and_crop_lines(image_path, crop_folder)

    results = []

    # Step 2: Process each line
    for path in line_paths:
        img = cv.imread(path)

        if not is_good_line(img):
            continue

        text = recognize_line(path)

        if not is_confident(text):
            continue

        results.append(text)

    # Step 3: Merge
    final_text = "\n".join(results)

    return final_text


# =========================
# RUN
# =========================
if __name__ == "__main__":
    image_path = r"C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset\scan1_page1.jpeg"

    text = run_ocr(image_path)

    print("\n===== OCR OUTPUT =====\n")
    print(text)