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
        binary = preprocess_img(img_path)  # calling the preprocess function to get the binary image
        if binary is None:
            return

        original = cv.imread(img_path)
        if original is None:
            raise ValueError(f"Original image load failed: {img_path}")

        inverted = cv.bitwise_not(binary)  # using bitwise not to invert blackt->white and white->black

        # find sum of pixels in each row
        row_sum = np.sum(inverted, axis=1)
        # print(f"Max sum: {row_sum.max()}")
        # print(f"Min sum: {row_sum.min()}")
        # print(f"Mean sum: {row_sum.mean()}")

        # find the indices of rows where the sum is greater than a threshold (indicating presence of text)
        threshold = int(row_sum.mean() * 0.5)

        line_indices = np.where(row_sum > threshold)[0]

        # group consecutive line indices together to form line segments
        lines = []
        if len(line_indices) > 0:
            start = line_indices[0]
            for i in range(1, len(line_indices)):
                # if gap between current and previous > 5 pixels
                if line_indices[i] - line_indices[i - 1] > 5:
                    lines.append((start, line_indices[i - 1]))  # group end
                    start = line_indices[i]  # naya group start
            lines.append((start, line_indices[-1]))  # last group

        # crop and save each line segment
        line = 1
        for (start_y, end_y) in lines:
            # add some padding to the line segment
            padding = 10
            start_y_padded = max(0, start_y - padding)
            end_y_padded = min(original.shape[0], end_y + padding)

            # crop from original image
            cropped = original[start_y_padded:end_y_padded, :]  # start_y to end_y, all columns (x-axis)

            # filter not very small lines (height > 10 pixels)
            if cropped.shape[0] > 10:  # height > 10 pixels
                save_path = os.path.join(output_folder, f'line_{page_number}_{line}.jpeg')

                success = cv.imwrite(save_path, cropped)
                if not success:
                    print(f"[ERROR] Failed to write image: {save_path}")

                line += 1

    except Exception as e:
        print(f"[ERROR] detect_and_crop_lines failed for {img_path}: {e}")


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
#     output_folder = r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\CroppedLines\test'
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     detect_and_crop_lines(test_image, output_folder, 'test')
#     print('Done! Check CroppedLines/test folder!')