import cv2
#cv2 for image processing
import os
#importing os for file handeling
from pdf2image import convert_from_path
#pdf2image module used as i had falsely made pdfs insted of jpeg files and this module converts pdfs to images

def pdf_to_img(pdf_path, output_folder,pdf_number):
    #a function made to conver pdfs into images
    images=convert_from_path(pdf_path=pdf_path,poppler_path=r'C:\Users\codes\poppler\Library\bin\poppler-25.12.0\Library\bin',dpi=300)
    #converting pdf to images using convert_from_path function of pdf2image module

    saved_paths=[]
    for i,image in enumerate(images):
        image_path=os.path.join(output_folder,f'scan{pdf_number}_page{i+1}.jpeg')
        image.save(image_path,'JPEG')
        saved_paths.append(image_path)
    print(f'{len(saved_paths)} images saved to {output_folder}')
    return saved_paths

if __name__=='__main__':
    output_folder=r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset'
    os.makedirs(output_folder, exist_ok=True)
    for i in range(1,150):
        pdf_path=rf"C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset_pdf\scan{i}.pdf"
        pdf_to_img(pdf_path,output_folder,pdf_number = i)