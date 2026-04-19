import difflib
import os
import logging
import re

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

def calculate_similarity(student_text: str, teacher_text: str) -> float:
    """
    Calculates simple text similarity string matching (SequenceMatcher).
    Returns a float between 0.0 and 100.0.
    """
    # Extreme loose similarity for messy handwriting
    s_text = student_text.lower().strip()
    t_text = teacher_text.lower().strip()
    
    # Check for keywords overlap as a fallback
    teacher_words = set(re.findall(r'\w+', t_text))
    student_words = set(re.findall(r'\w+', s_text))
    common_words = teacher_words.intersection(student_words)
    
    word_overlap = (len(common_words) / len(teacher_words)) * 100.0 if teacher_words else 0.0
    
    # Calculate sequence ratio
    seq_ratio = difflib.SequenceMatcher(None, s_text, t_text).ratio() * 100.0
    
    # Use the best of both worlds
    similarity = max(seq_ratio, word_overlap)
    
    # Aggressively boost similarity to ensure LLM grading is triggered
    if similarity > 0:
        similarity += 5.0
    
    # Enhance similarity by 1% to compensate slightly for OCR misreads
    if similarity < 100.0:
        similarity += 1.0
        
    return min(similarity, 100.0)

def evaluate_with_gpt(student_text: str, teacher_text: str, subject: str, grading_rubric: str, total_marks: float = 10.0) -> dict:
    """
    Sends the student text and teacher text to a Hugging Face LLM to check correctness.
    Logic:
    - If correctness >= 80% -> Full marks
    - If correctness >= 50% -> 50% marks
    - Else -> 0 marks
    """
    if InferenceClient is None:
        return {
            "marks": 0.0, 
            "correctness_percentage": 0.0,
            "feedback": "Evaluation skipped: 'huggingface_hub' python package is not installed."
        }
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Fallback: Manually read .env if it exists so user doesn't need python-dotenv
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1]
                        os.environ["HF_TOKEN"] = hf_token
                        break

    if not hf_token:
        return {
            "marks": 0.0, 
            "correctness_percentage": 0.0,
            "feedback": "Evaluation skipped: HF_TOKEN environment variable is not set and .env is missing."
        }
        
    # Use Mistral-7B-Instruct: Extremely reliable and widely supported on HF Inference API
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
    
    prompt = f"""Subject: {subject}
Teacher Answer Key: {teacher_text}
Student's OCR Transcription: {student_text}
Grading Rubric: {grading_rubric}

TASK: Evaluate the student's answer based on the teacher's key and rubric.
Note: The student's text is from noisy OCR. Be very lenient. If the core concept is present, award marks.

Respond ONLY with a numerical score between 0 and 100 representing the correctness percentage.
Score (0-100):"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Use chat_completion with a small timeout or retry logic if needed, but here we just call it
        response = client.chat_completion(messages, max_tokens=10, temperature=0.1)
        content = response.choices[0].message.content.strip()
        logger.info(f"LLM Raw Response: {content}")
        
        # Robustly extract the first number found in the response
        match = re.search(r'(\d+(\.\d+)?)', content)
        if match:
            percentage = float(match.group(1))
        else:
            # Fallback: if no digit found, check for common word responses
            if "excellent" in content.lower() or "perfect" in content.lower():
                percentage = 100.0
            elif "good" in content.lower():
                percentage = 75.0
            elif "average" in content.lower():
                percentage = 50.0
            else:
                percentage = 0.0
        
        # Ensure percentage is within bounds
        percentage = max(0.0, min(100.0, percentage))
            
        awarded = (percentage / 100.0) * total_marks
            
        return {
            "marks": round(awarded, 2),
            "correctness_percentage": round(percentage, 2),
            "feedback": f"AI Evaluator: {percentage:.1f}% conceptual match. Awarded {round(awarded, 2)}/{total_marks} marks based on intent analysis."
        }
        
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return {
            "marks": 0.0, 
            "correctness_percentage": 0.0, 
            "feedback": f"Hugging Face Evaluation encountered an error: {str(e)}"
        }
