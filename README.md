# 🚀 Exam Lense AI

Exam Lense AI is an **AI-powered system for reading and evaluating handwritten answer sheets** using a fine-tuned TrOCR model.

It is designed to **understand real student handwriting**, extract the exact written content, and generate meaningful evaluation results — making the process faster, smarter, and more reliable.

---

## 🔥 What Makes It Different?

Most AI systems (like ChatGPT) work by:

* Predicting the **next probable word (token)**
* Auto-correcting or adjusting input text
* Sometimes **changing the original meaning**

👉 This leads to **loss of original handwritten data**

---

### 🧠 Exam Lense Approach

Exam Lense is built differently:

* Focuses on **reading exactly what is written**
* Preserves **original handwriting content**
* Minimizes **data loss and over-correction**
* Designed specifically for **real exam answer sheets**

---

## ✍️ Custom Model

* Fine-tuned by **Aditya Prasad Barik**
* Trained on a **custom dataset**
* Specialized for **Indian cursive handwriting**
* Built to handle **real-world student writing styles**

This makes it a **highly practical and targeted OCR system**, not just a general AI model.

---

## 📊 Performance

* ✅ Accuracy: **~85%**
* 🎯 Works best on:

  * Clear handwritten answers
  * Structured exam sheets

---

## ⚠️ Limitations

* May struggle with **very poor or unclear handwriting**
* Can occasionally **hallucinate characters in noisy input**
* Accuracy depends on **image quality and writing clarity**

---

## 💡 What It Does

* Reads handwritten answer sheets
* Extracts text line-by-line
* Helps in automated evaluation

---

## 🧱 Tech Stack

* **Python** – Core backend language
* **FastAPI** – Backend framework
* **PyTorch** – Model training & inference
* **Transformers (Hugging Face)** – TrOCR model
* **OpenCV** – Image preprocessing
* **NumPy / Pandas** – Data handling

---

## 📁 Project Structure

```bash
exam-lense/
├── app/
│   ├── main.py
│   ├── api/              # API routes
│   ├── core/             # Config & logging
│   ├── models/           # Fine-tuned TrOCR model
│   ├── services/         # OCR + evaluation logic
│   ├── utils/            # Image processing helpers
│   └── schemas/          # Data schemas
├── data/                 # Dataset & labels
├── uploads/              # Input/output images
├── scripts/              # Utility scripts
├── tests/                # Testing
├── requirements.txt
└── README.md
```

---

## 🚀 Vision

To build a system that can:

* Digitize handwritten exams
* Reduce manual checking effort
* Enable smarter and faster evaluation

---

## 👨‍💻 Author

**Aditya Prasad Barik**

---

## 📜 License

MIT License
