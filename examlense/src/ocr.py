# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# # import requests if we will take images from the web

# def load_model():
#     processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
#     model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
#     return processor, model

# #image=Image.open(requests.get('https://raw.githubusercontent.com/huggingface/notebooks/main/examples/datasets/ocr/handwritten.png', stream=True).raw).convert('RGB')
# #This can be used if we want to take images from the web, but for now we will use local images
# def extract_text(img_path,processor, model):
#     image=Image.open(img_path).convert('RGB')
#     pixel_values=processor(images=image,return_tensors='pt').pixel_values
#     generated_ids=model.generate(pixel_values,max_new_tokens=51)
#     text=processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
#     return text

# #for testing purposes
# if __name__== '__main__':
#     processor, model = load_model()
#     text = extract_text(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\preprocessed_test2.jpeg', processor, model)
#     print(text)

''''This code failed as indian cursive letterers are tough to be recognized by the model,
so we will use easyocr which is more robust and can handle a variety of fonts and handwriting styles.
And wait for this apks version 2.0 which will be made after fine tuning the TrOCR model on indian cursive letters.'''



# import easyocr

# def load_model():
#     reader = easyocr.Reader(['en'])
#     return reader

# def extract_text(img_path, reader):
#     result = reader.readtext(img_path, detail=1, paragraph=False)
#     # sirf text part nikalo, confidence 0.3 se upar wale
#     lines = [item[1] for item in result if item[2] > 0.3]
#     return '\n'.join(lines)

# if __name__ == '__main__':
#     reader = load_model()
#     text = extract_text(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\preprocessed_test2.jpeg', reader)
#     print(text)

''''This code failed as indian cursive letterers are tough to be recognized by the model,
so we will use google cloud vision api which is more robust and can handle a variety of fonts and handwriting styles.'''

import os
from google.cloud import vision

# credentials set karo
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\codes\Desktop\Exam lense\examlense\credentials.json'

def load_model():
    client = vision.ImageAnnotatorClient()
    return client

def extract_text(img_path, client):
    with open(img_path, 'rb') as f:
        content = f.read()
    
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text
    return text

if __name__ == '__main__':
    client = load_model()
    text = extract_text(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\preprocessed_test2.jpeg', client)
    print(text)