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

    try:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Image load failed:{img_path}")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Clean binarization - step1
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        binary = cv.bitwise_not(binary)

        # ignoring notebook horizontal lines - step2
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 1))
        lines = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

        clean = cv.subtract(binary, lines)

        # Horizontal projection fixing - Step 3
        projection = np.sum(clean, axis=1)

        # Smoothening projection -step 4
        projection = cv.GaussianBlur(projection.reshape(-1, 1), (15, 1), 0).flatten()

        # Normalizing the binary projection - step 5
        projection = projection / np.max(projection)

        # finding valleys step 6
        threshold = 0.15   #logic here is  LOW = detect gaps

        separators = np.where(projection < threshold)[0]

        #group separators into regions - step 7:
        gaps = []
        if len(separators) > 0:
            start = separators[0]

            for i in range(1, len(separators)):
                if separators[i] - separators[i - 1] > 5:
                    gaps.append((start, separators[i - 1]))
                    start = separators[i]

            gaps.append((start, separators[-1]))

        #convert gaps → line regions - Step 8
        lines = []
        prev_end = 0

        for (g_start, g_end) in gaps:
            if g_start - prev_end > 10:
                lines.append((prev_end, g_start))
            prev_end = g_end

        if prev_end < img.shape[0]:
            lines.append((prev_end, img.shape[0]))

        #Smart filtering of lines - remove too small or too large lines - step 9
        final_lines = []
        heights = [end - start for (start, end) in lines]

        if heights:
            avg_h = np.mean(heights)

            for (start, end) in lines:
                h = end - start

                if h < avg_h * 0.5:
                    continue
                if h > avg_h * 1.5:
                    continue

                final_lines.append((start, end))

        # Crop with SAFE padding - Step 10
        line_no = 1

        for (y1, y2) in final_lines:
            pad = int((y2 - y1) * 0.25)

            y1 = max(0, y1 - pad)
            y2 = min(img.shape[0], y2 + pad)

            crop = img[y1:y2, :]

            if crop.shape[0] > 20:
                save_path = os.path.join(
                    output_folder,
                    f'line_{page_number}_{line_no}.jpeg'
                )
                cv.imwrite(save_path, crop)
                line_no += 1

    except Exception as e:
        print(f"[ERROR] {e}")






# if __name__ == '__main__':
#     input_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset'
#     output_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines'

#     try:
#         os.makedirs(output_folder, exist_ok=True)

#         if not os.path.exists(input_folder):
#             raise FileNotFoundError(f"Input folder not found: {input_folder}")

#         for filename in os.listdir(input_folder):
#             if filename.lower().endswith('.jpeg'):
#                 img_path = os.path.join(input_folder, filename)

#                 if not os.path.isfile(img_path):
#                     continue

#                 # scan1_page1.jpeg → line_scan1_page1
#                 page_number = filename.replace('.jpeg', '')

#                 detect_and_crop_lines(img_path, output_folder, page_number)
#                 print(f'{filename} processed!')

#         print('ALL DONE! Check CroppedLines folder!')

#     except Exception as e:
#         print(f"[FATAL ERROR] {e}")


# successfully tested on one image, now let's run on the whole dataset
if __name__ == '__main__':
    # test for a single image
    test_image = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset\scan1_page1.jpeg'
    output_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines\test2'
    
    os.makedirs(output_folder, exist_ok=True)
    
    detect_and_crop_lines(test_image, output_folder, 'test')
    print('Done! Check CroppedLines/test folder!')