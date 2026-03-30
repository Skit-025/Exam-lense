import cv2 as cv
import numpy as np
import os


def preprocess_img(img_path):
    try:
        img = cv.imread(img_path)  # reads the image using opencv
        if img is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")

        # let's convert that BGR image into a grayscale image
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # let's apply some gaussian blur(smoother texture)
        imgray = cv.GaussianBlur(imgray, (5, 5), 0)

        # apply adaaptive brightness to get the binary image
        imbinary = cv.adaptiveThreshold(
            imgray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            31,
            15
        )

        return imbinary

    except Exception as e:
        print(f"[ERROR] preprocess_img failed for {img_path}: {e}")
        return None


def detect_and_crop_lines(img_path, output_folder, page_number):
    import cv2 as cv
    import numpy as np
    import os

    try:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Image load failed: {img_path}")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # --- Step 1: Strong binarization ---
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        inverted = cv.bitwise_not(binary)

        # --- Step 2: Remove notebook horizontal lines ---
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        removed_lines = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel)

        clean = cv.subtract(inverted, removed_lines)

        # --- Step 3: Row projection ---
        row_sum = np.sum(clean, axis=1)

        # Smooth the signal
        row_sum = cv.GaussianBlur(row_sum.reshape(-1, 1), (7, 1), 0).flatten()

        # Dynamic threshold
        threshold = np.percentile(row_sum, 65)

        line_indices = np.where(row_sum > threshold)[0]

        # --- Step 4: Group lines ---
        lines = []
        if len(line_indices) > 0:
            gap = int(img.shape[0] * 0.012)  # adaptive gap
            start = line_indices[0]

            for i in range(1, len(line_indices)):
                if line_indices[i] - line_indices[i - 1] > gap:
                    lines.append((start, line_indices[i - 1]))
                    start = line_indices[i]

            lines.append((start, line_indices[-1]))

        # --- Step 5: Smart filtering (VERY IMPORTANT) ---
        final_lines = []
        heights = [end - start for (start, end) in lines]

        if len(heights) > 0:
            avg_height = np.mean(heights)

            for (start, end) in lines:
                h = end - start

                # Reject too small (noise)
                if h < avg_height * 0.5:
                    continue

                # Reject too big (merged lines)
                if h > avg_height * 1.8:
                    continue

                final_lines.append((start, end))

        # --- Step 6: Crop safely ---
        line_no = 1
        for (start_y, end_y) in final_lines:
            padding = int((end_y - start_y) * 0.3)

            start_y = max(0, start_y - padding)
            end_y = min(img.shape[0], end_y + padding)

            cropped = img[start_y:end_y, :]

            # FINAL CHECK → avoid empty crops
            if cropped.shape[0] > 15:
                save_path = os.path.join(output_folder, f'line_{page_number}_{line_no}.jpeg')
                cv.imwrite(save_path, cropped)
                line_no += 1

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    input_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset'
    output_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines'

    try:
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.jpeg'):
                img_path = os.path.join(input_folder, filename)

                if not os.path.isfile(img_path):
                    continue

                # scan1_page1.jpeg → line_scan1_page1
                page_number = filename.replace('.jpeg', '')

                detect_and_crop_lines(img_path, output_folder, page_number)
                print(f'{filename} processed!')

        print('ALL DONE! Check CroppedLines folder!')

    except Exception as e:
        print(f"[FATAL ERROR] {e}")


#successfully tested on one image, now let's run on the whole dataset
# if __name__ == '__main__':
#     # test for a single image
#     test_image = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset\scan1_page1.jpeg'
#     output_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines\test2'
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     detect_and_crop_lines_best(test_image, output_folder, 'test')
#     print('Done! Check CroppedLines/test folder!')