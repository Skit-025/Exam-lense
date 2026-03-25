from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
# import requests if we will take images from the web

def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

#image=Image.open(requests.get('https://raw.githubusercontent.com/huggingface/notebooks/main/examples/datasets/ocr/handwritten.png', stream=True).raw).convert('RGB')
#This can be used if we want to take images from the web, but for now we will use local images
def extract_text(img_path,processor, model):
    image=Image.open(img_path).convert('RGB')
    pixel_values=processor(images=image,return_tensors='pt').pixel_values
    generated_ids=model.generate(pixel_values)
    text=processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
    return text

#for testing purposes
if __name__== '__main__':
    processor, model = load_model()
    text = extract_text(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\test.jpeg', processor, model)
    print(text)