import cv2 as cv

def preprocess_image(image_path):
    # read the given coloured image
    img=cv.imread(image_path)

    # converting that rgb image into grayscale img
    imgray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# applying gaussian blur to remove noise and smoothen the image
    imgray=cv.GaussianBlur(imgray,(5,5),0)

    #applying adaptive thresholding to get the binary image
    imbinary=cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,15)
    return imbinary

if __name__ == '__main__':
    preprocessed_image = preprocess_image(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\test.jpeg')
    cv.imwrite(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\preprocessed_test2.jpeg', preprocessed_image)