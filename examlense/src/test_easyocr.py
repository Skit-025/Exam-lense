import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext(r'C:\Users\codes\Desktop\Exam lense\examlense\uploads\.jpeg', detail=0)
print(' '.join(result))