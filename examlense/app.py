import os
import shutil
import uuid
import logging
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- User's Preexisting Imports ---
# We use your preexisting OCR pipeline directly so Microsoft TrOCR logic kicks in untouched.
try:
    from tools.ocr_pipeline import run_ocr
except ImportError:
    run_ocr = None
    
# Import the newly implemented Evaluator module
try:
    from src.evaluator import calculate_similarity, evaluate_with_gpt
except ImportError:
    calculate_similarity = None
    evaluate_with_gpt = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exam Lense AI Backend",
    description="End to End Evaluation Pipeline reading N pages and grading them via GPT.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class EvaluationResult(BaseModel):
    total_pages_processed: int
    combined_extracted_text: str
    similarity_percentage: float
    gpt_marks: float
    gpt_correctness_percentage: float
    gpt_feedback: str
    status: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Exam Lense AI End-to-End Pipeline is running."}

@app.post("/api/v1/evaluate", response_model=EvaluationResult)
def evaluate_answers(
    files: List[UploadFile] = File(...),
    student_name: str = Form(...),
    teacher_answer: str = Form(...),
    subject: str = Form(...),
    grading_rubric: str = Form(...),
    total_marks: float = Form(10.0)
):
    """
    1. Accepts N number of pages (files).
    2. Processes them using your pre-existing TrOCR model (breaking into single cropped lines).
    3. Combines the text from all pages.
    4. Computes similarity with the provided teacher answer sheet.
    5. If similarity > 40%, it evaluates using GPT.
    6. Grades exactly 80% correctness = full marks, 50% = half, else 0.
    """
    try:
        if not run_ocr:
            raise HTTPException(status_code=500, detail="tools.ocr_pipeline is missing or failed to import.")
        if not calculate_similarity or not evaluate_with_gpt:
            raise HTTPException(status_code=500, detail="src.evaluator module is missing or failed to import.")

        combined_text_chunks = []
        
        for file in files:
            # Fix: Gradio/Requests often send images as 'application/octet-stream' or generic types
            # We explicitly check extensions or just allow standard image stream types
            is_image = file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
            
            if not is_image:
                logger.warning(f"Skipping file {file.filename} due to invalid content-type: {file.content_type}")
                continue
                
            file_ext = file.filename.split(".")[-1]
            unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                logger.info(f"Running preexisting Microsoft Base Model pipeline for: {file.filename}")
                extracted_page_text = run_ocr(file_path)
                
                if extracted_page_text:
                    combined_text_chunks.append(extracted_page_text)
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
            finally:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                        
        final_combined_student_text = "\n".join(combined_text_chunks)
        
        y_pages = len(files)
        
        if not final_combined_student_text.strip():
            return EvaluationResult(
                total_pages_processed=y_pages,
                combined_extracted_text="",
                similarity_percentage=0.0,
                gpt_marks=0.0,
                gpt_correctness_percentage=0.0,
                gpt_feedback=f"ALERT: The {y_pages} page(s) of student '{student_name}' could not be read at all by the OCR model.",
                status="failed"
            )
            
        similarity = calculate_similarity(final_combined_student_text, teacher_answer)
        logger.info(f"Similarity mapping to expected answer: {similarity:.2f}%")
        
        # BOUNCE BACK LOGIC: If similarity is terribly low, it means hallucination / poor handwriting
        if similarity <= 5.0:
            return EvaluationResult(
                total_pages_processed=y_pages,
                combined_extracted_text=final_combined_student_text,
                similarity_percentage=similarity,
                gpt_marks=0.0,
                gpt_correctness_percentage=0.0,
                gpt_feedback=f"🚨 ALERT: The {y_pages} page(s) of student '{student_name}' cannot be automatically evaluated. The OCR model is hallucinating heavily or the handwriting is too poor. Please go for MANUAL CORRECTION.",
                status="manual_correction_required"
            )
        
        gpt_result = {
            "marks": 0.0,
            "correctness_percentage": 0.0,
            "feedback": "Similarity was critically low. GPT verification was bypassed."
        }
        
        if similarity > 5.0:
            logger.info("Similarity > 5%. Passing onto GPT for deep semantic verification...")
            gpt_result = evaluate_with_gpt(final_combined_student_text, teacher_answer, subject=subject, grading_rubric=grading_rubric, total_marks=total_marks)

        return EvaluationResult(
            total_pages_processed=y_pages,
            combined_extracted_text=final_combined_student_text,
            similarity_percentage=similarity,
            gpt_marks=gpt_result.get("marks", 0.0),
            gpt_correctness_percentage=gpt_result.get("correctness_percentage", 0.0),
            gpt_feedback=gpt_result.get("feedback", ""),
            status="success"
        )
    except Exception as general_error:
        logger.error(f"CRITICAL 500 ERROR: {general_error}")
        return EvaluationResult(
            total_pages_processed=0,
            combined_extracted_text="",
            similarity_percentage=0.0,
            gpt_marks=0.0,
            gpt_correctness_percentage=0.0,
            gpt_feedback=f"CRITICAL BACKEND ERROR: {str(general_error)}",
            status="error_500"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
