import cv2 as cv
import numpy as np
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
from PIL import Image

# =========================
# LAZY LOAD MODEL
# =========================
processor = None
model = None

def get_trocr_model():
    global processor, model
    if processor is None or model is None:
        custom_lora_path = r"C:\Users\codes\Desktop\final_model_lora"
        # FORCE: Your LoRA checkpoint is 768-size. We MUST use TrOCR-Base.
        base_model_name = "microsoft/trocr-base-handwritten"

        print(f"Loading Base Foundation: {base_model_name}")
        
        processor = TrOCRProcessor.from_pretrained(base_model_name)
        base_model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
        
        print(f"DEBUG: Model Hidden Size is {base_model.config.decoder.hidden_size}")

        if PeftModel and os.path.exists(custom_lora_path):
            print(f"Applying Fine-Tuned LoRA Adaptors from: {custom_lora_path}")
            model = PeftModel.from_pretrained(base_model, custom_lora_path)
            model = model.merge_and_unload()
            print("LoRA Integration Successful.")
        else:
            model = base_model

    return processor, model


# =========================
# PREPROCESSING (LEGACY)
# =========================
def preprocess_img(img_path):
    try:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgray = cv.GaussianBlur(imgray, (5, 5), 0)
        imbinary = cv.adaptiveThreshold(
            imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15
        )
        return imbinary
    except Exception as e:
        print(f"[ERROR] preprocess_img failed: {e}")
        return None

# =========================
# LINE SEGMENTATION (USER-APPROVED HIGH PRECISION)
# =========================
def detect_and_crop_lines(img_path, output_folder, page_number="page"):
    try:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Image load failed: {img_path}")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        binary = cv.bitwise_not(binary)

        # Ignore notebook lines
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 1))
        lines_morph = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        clean = cv.subtract(binary, lines_morph)

        # Horizontal projection
        projection = np.sum(clean, axis=1)
        projection = cv.GaussianBlur(projection.reshape(-1, 1), (15, 1), 0).flatten()
        
        # Avoid division by zero
        if np.max(projection) == 0: return []
        projection = projection / np.max(projection)

        # Find valleys (gaps)
        threshold = 0.15 
        separators = np.where(projection < threshold)[0]

        gaps = []
        if len(separators) > 0:
            start = separators[0]
            for i in range(1, len(separators)):
                if separators[i] - separators[i - 1] > 5:
                    gaps.append((start, separators[i - 1]))
                    start = separators[i]
            gaps.append((start, separators[-1]))

        # Convert gaps to line regions
        lines = []
        prev_end = 0
        for (g_start, g_end) in gaps:
            if g_start - prev_end > 10:
                lines.append((prev_end, g_start))
            prev_end = g_end

        if prev_end < img.shape[0]:
            lines.append((prev_end, img.shape[0]))

        # Smart filtering
        final_lines = []
        heights = [end - start for (start, end) in lines]
        if heights:
            avg_h = np.mean(heights)
            for (start, end) in lines:
                h = end - start
                if h < avg_h * 0.5 or h > avg_h * 1.5: continue
                final_lines.append((start, end))

        # Crop and Save
        os.makedirs(output_folder, exist_ok=True)
        paths = []
        for i, (y1, y2) in enumerate(final_lines):
            pad = int((y2 - y1) * 0.25)
            y1_p = max(0, y1 - pad)
            y2_p = min(img.shape[0], y2 + pad)
            crop = img[y1_p:y2_p, :]

            if crop.shape[0] > 20:
                save_path = os.path.join(output_folder, f'line_{page_number}_{i+1}.jpeg')
                cv.imwrite(save_path, crop)
                paths.append(save_path)
        
        return paths

    except Exception as e:
        print(f"[ERROR] Segmentation failed: {e}")
        return []


# =========================
# QUALITY CHECK
# =========================
def is_good_line(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    non_zero = np.count_nonzero(gray < 230)
    ratio = non_zero / (img.shape[0] * img.shape[1])

    if ratio < 0.005:
        return False

    h = img.shape[0]
    if h < 5 or h > 300:
        return False

    return True

# =========================
# OCR (TrOCR)
# =========================
def recognize_line(img_path):
    proc, mod = get_trocr_model()
    
    # EMERGENCY SHARPENING BOOSTER
    img_cv = cv.imread(img_path)
    if img_cv is not None:
        # Step 1: Grayscale & Denoise
        gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)
        
        # Step 2: High-Pass Sharpening Filter
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv.filter2D(gray, -1, sharpen_kernel)
        
        # Step 3: Adaptive Binarization (Makes text "Pop")
        _, thresh = cv.threshold(sharpened, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Convert back to PIL for Model
        image = Image.fromarray(thresh).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    pixel_values = proc(images=image, return_tensors="pt").pixel_values
    generated_ids = mod.generate(pixel_values, max_new_tokens=256)

    text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# =========================
# CONFIDENCE FILTER
# =========================
def is_confident(text):
    if not text.strip():
        return False
    return True


# =========================
# MAIN PIPELINE
# =========================
def run_ocr(image_path):
    crop_folder = os.path.join(os.getcwd(), "temp_lines")
    os.makedirs(crop_folder, exist_ok=True)

    # Step 1: Segment
    page_name = os.path.basename(image_path).split('.')[0]
    line_paths = detect_and_crop_lines(image_path, crop_folder, page_name)
    
    if not line_paths:
        print("[WARNING] No lines detected. Falling back to whole-page OCR.")
        line_paths = [image_path]

    results = []

    # Step 2: Process each line
    for path in line_paths:
        img = cv.imread(path)
        if img is None: continue

        # Skip quality checks for whole-page fallback or small segments
        if len(line_paths) > 1 and not is_good_line(img):
            continue

        text = recognize_line(path)
        if is_confident(text):
            results.append(text)

    # Step 3: Merge
    final_text = " ".join(results)
    return final_text.strip()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    for i in range(1,26):
        if i==3 or i==5 or i==7:
            continue
        image_path = rf"C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines\line_scan1_page1_{i}.jpeg"
        # print(os.path.exists(image_path))

        text = run_ocr(image_path)

        print("\n===== OCR OUTPUT =====\n")
        print(text)
        # counter=16