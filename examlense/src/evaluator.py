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
        
    # Use Google Gemma-2-9B: Advanced instruction following
    client = InferenceClient("google/gemma-2-9b-it", token=hf_token)
    
    messages = [
        {"role": "user", "content": f"""Subject: {subject}
Rules: {grading_rubric}
Teacher Correct Answer: {teacher_text}
Student Answer (Extracted via OCR): {student_text}

TASK: Evaluate the student's marks (0-100%). 
CRITICAL NOTE: The student answer was read via an OCR model and may have minor spelling mistakes or noise (e.g. '0' instead of 'o'). 
Focus on the CONCEPTUAL intent. If the intent matches the teacher's key despite OCR typos, award the marks.
Respond with ONLY the number (0-100)."""}
    ]
    
    try:
        # Use chat_completion for robust task-routing on HuggingFace
        response = client.chat_completion(messages, max_tokens=10, temperature=0.1)
        content = response.choices[0].message.content.strip()
        
        # Extract the number from the response
        match = re.search(r'\d+(\.\d+)?', content)
        percentage = float(match.group()) if match else 0.0
        
        # Strict Marking Logic Implementation
        # 1. No word matches / zero percentage -> Zero
        if percentage <= 0:
            awarded = 0.0
        else:
            # 2. Linear calculation based on AI percentage
            awarded = (percentage / 100.0) * total_marks
            
        # 3. Floor Logic: If final mark is < 1, it becomes zero
        if awarded < 1.0:
            awarded = 0.0
            
        return {
            "marks": awarded,
            "correctness_percentage": percentage,
            "feedback": f"AI calculated {percentage:.1f}% correctness. Final mark awarded: {awarded} (Minimum 1.0 mark required for credit)."
        }
        
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return {
            "marks": 0.0, 
            "correctness_percentage": 0.0, 
            "feedback": f"Hugging Face Evaluation encountered an error: {str(e)}"
        }
