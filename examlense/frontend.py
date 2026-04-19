import gradio as gr
import urllib.request
import xml.etree.ElementTree as ET
import random
import os
import time
import shutil

# Direct Backend Imports for Maximum Stability
try:
    from tools.ocr_pipeline import run_ocr
    from src.evaluator import calculate_similarity, evaluate_with_gpt
except ImportError:
    run_ocr = None
    calculate_similarity = None
    evaluate_with_gpt = None

def fetch_rss_items(query_url, limit):
    try:
        req = urllib.request.Request(query_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        items = []
        for item in root.findall('.//item')[:limit]:
            title = item.find('title').text
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            link = item.find('link').text
            items.append((title, link))
        return items
    except Exception:
        return []

def fetch_education_news():
    url_world = "https://news.google.com/rss/search?q=global+education+schools+policy&hl=en-US&gl=US&ceid=US:en"
    url_india = "https://news.google.com/rss/search?q=education+schools+india&hl=en-IN&gl=IN&ceid=IN:en"
    url_odisha = "https://news.google.com/rss/search?q=education+schools+odisha&hl=en-IN&gl=IN&ceid=IN:en"
    
    world_news = fetch_rss_items(url_world, 4)
    india_news = fetch_rss_items(url_india, 4)
    odisha_news = fetch_rss_items(url_odisha, 2)
    
    all_news = world_news + india_news + odisha_news
    random.shuffle(all_news)
    
    if not all_news:
        return "<div style='margin-top: 30px; color: #fca5a5;'>Unable to load live news right now.</div>"

    news_html = "<div style='margin-top: 30px; padding: 20px; background: rgba(4, 47, 46, 0.4); border-radius: 12px; border: 1px solid rgba(212, 175, 55, 0.2); box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>"
    news_html += "<h2 style='color: #d4af37; margin-bottom: 15px; text-align: center; border-bottom: 1px solid rgba(212, 175, 55, 0.2); padding-bottom: 10px; font-family: Playfair Display;'>🌍 Live Global & Regional Education Updates</h2>"
    news_html += "<ul style='list-style-type: none; padding: 0; margin: 0;'>"
    
    for title, link in all_news:
        news_html += f"<li style='margin-bottom: 14px; font-size: 0.95rem; line-height: 1.4;'><a href='{link}' target='_blank' style='color: #a7f3d0; text-decoration: none;' onmouseover=\"this.style.color='#fef08a'\" onmouseout=\"this.style.color='#a7f3d0'\">➤ {title}</a></li>"
    
    news_html += "</ul></div>"
    return news_html

def run_evaluation(binary_files, student_name, subject, grading_rubric, teacher_answer, total_marks, progress=gr.Progress()):
    print(f"\n[DEBUG] run_evaluation triggered for {student_name}!")
    start_time = time.time()
    expected_time = len(binary_files) * 15 if binary_files else 15
    
    yield "⏳ Initializing Engine...", f"### 📊 Preparation Phase\nExpected Processing Time: {expected_time}s\nElapsed: 0s"

    if not binary_files:
        yield "❌ Empty Upload Detected", "No images detected in the upload payload (Gradio returned None)."
        return

    # Save binaries to local disk manually to bypass Gradio temp-file locking
    local_paths = []
    temp_dir = os.path.join(os.getcwd(), "manual_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        progress(0.1, desc="Localization...")
        # Handle single or multiple file binaries
        file_list = binary_files if isinstance(binary_files, list) else [binary_files]
        
        for i, f_data in enumerate(file_list):
            try:
                if isinstance(f_data, bytes):
                    f_path = os.path.join(temp_dir, f"page_{i}_{int(time.time())}.jpg")
                    with open(f_path, "wb") as f:
                        f.write(f_data)
                    local_paths.append(f_path)
                elif isinstance(f_data, dict) and "data" in f_data:
                    f_path = os.path.join(temp_dir, f"page_{i}_{int(time.time())}.jpg")
                    with open(f_path, "wb") as f:
                        f.write(f_data["data"])
                    local_paths.append(f_path)
                elif hasattr(f_data, "name"):
                    local_paths.append(f_data.name)
                elif isinstance(f_data, str):
                    local_paths.append(f_data)
            except Exception as fe:
                print(f"[ERROR] Failed to save binary chunk: {fe}")

        if not local_paths:
            yield "❌ Storage Failure", "Images were uploaded but could not be saved to your local disk."
            return

        combined_text_chunks = []
        for i, filepath in enumerate(local_paths):
            elapsed = int(time.time() - start_time)
            msg = f"Reading Page {i+1} of {len(local_paths)}..."
            progress((i/len(local_paths))*0.6, desc=msg)
            yield f"📖 {msg}", f"### 🤖 AI Vision Active\n- **Current Task:** OCR Segmentation\n- **Expected:** ~{expected_time}s\n- **Elapsed:** {elapsed}s"
            
            text = run_ocr(filepath)
            if text:
                combined_text_chunks.append(text)
                
        final_student_text = "\n".join(combined_text_chunks)
        y_pages = len(local_paths)

        if not final_student_text.strip():
            yield "❌ OCR Failed", f"The {y_pages} page(s) could not be read. Ensure the handwriting is visible."
            return

        progress(0.8, desc="AI Semantic Grading...")
        elapsed = int(time.time() - start_time)
        yield "🧠 Analyzing Context...", f"### 📊 Grading Phase\n- **OCR Completion:** 100%\n- **Expected:** ~{expected_time}s\n- **Elapsed:** {elapsed}s"
        
        similarity = calculate_similarity(final_student_text, teacher_answer)
        
        # PRO-TEACHER UPDATE: Remove gate to ensure conceptual evaluation always occurs
        print(f"[DEBUG] Similarity: {similarity:.1f}%. Triggering conceptual AI grader...")
        gpt_result = evaluate_with_gpt(final_student_text, teacher_answer, subject, grading_rubric, float(total_marks))

        progress(1.0, desc="Finalizing Report...")
        elapsed = int(time.time() - start_time)
        output_status = f"✅ Success - Evaluated in {elapsed}s."
        formatted_res = f"### 📊 Evaluation Result\n"
        formatted_res += f"- **Mark:** {gpt_result.get('marks')} / {total_marks}\n"
        formatted_res += f"- **Correctness:** {gpt_result.get('correctness_percentage')}%\n"
        formatted_res += f"- **Feedback:** {gpt_result.get('feedback')}\n\n"
        formatted_res += f"### 📑 OCR Transcription\n```text\n{final_student_text}\n```"
        
        yield output_status, formatted_res

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        yield "Local Crash", str(e)
        return

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800&family=Manrope:wght@300;400;600&display=swap');

footer {
    display: none !important;
}

body, .gradio-container {
    font-family: 'Manrope', sans-serif !important;
    background: radial-gradient(circle at top right, #064e3b 0%, #020617 100%) !important;
    color: #f8fafc !important;
}
.gr-box, .gr-panel, .gradio-container-3-50 {
    background: rgba(4, 47, 46, 0.3) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
}
textarea, input[type="text"], input[type="number"] {
    background: rgba(2, 6, 23, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 14.5px !important;
    transition: all 0.3s ease !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: #d4af37 !important;
    box-shadow: 0 0 12px rgba(212, 175, 55, 0.3) !important;
    outline: none !important;
}
.status-alive {
    border-left: 4px solid #d4af37 !important;
}
.gr-button-primary {
    font-family: 'Playfair Display', serif !important;
    background: linear-gradient(135deg, #d4af37 0%, #b48600 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #020617 !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 20px rgba(212, 175, 55, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
}
.gr-button-primary:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 30px rgba(212, 175, 55, 0.5) !important;
    background: linear-gradient(135deg, #fbbf24 0%, #d4af37 100%) !important;
}
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 800 !important;
    background: -webkit-linear-gradient(45deg, #fef08a, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem !important;
    text-align: center;
    margin-top: 15px !important;
    margin-bottom: 0.5rem !important;
}
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #fef08a !important;
    font-weight: 700 !important;
}
label, .gr-form > label {
    font-family: 'Manrope', sans-serif !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
}
"""

with gr.Blocks(title="Exam Lense Evaluation") as demo:
    images_state = gr.State([])
    gr.Markdown("# 🏛️ Exam Lense - Classical Evaluation Hub")
    gr.Markdown("<p style='text-align: center; color: #a7f3d0; font-size: 1.15rem; margin-bottom: 35px; font-weight: 300;'>Upload student pages, apply dynamic grading matrices, and view automated scoring matrices instantly.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            student_name_input = gr.Textbox(label="Student Name / Roll No.", placeholder="e.g. Aditya Barik")
            subject_input = gr.Textbox(label="Exam Subject", placeholder="e.g. Computer Science")
            total_marks_input = gr.Number(label="Total Marks Available", value=10.0, step=0.5)
            
            grading_rubric_input = gr.TextArea(
                label="Grading Rubric / Rules", 
                placeholder="e.g. 1-10 number 10 marks, 10-29 4 marks each...",
                lines=3
            )
            
            teacher_key_input = gr.TextArea(
                label="Teacher's Answer Key", 
                placeholder="Paste the official answers here...",
                lines=5
            )
            
            images_input = gr.File(
                label="Step 1: Upload Student Exam Pages (Images)", 
                file_count="multiple",
                type="binary",
                interactive=True
            )
            
            evaluate_btn = gr.Button("🚀 🧠 Evaluate Student Answer", variant="primary")
            
        with gr.Column(scale=1):
            status_output = gr.Textbox(label="System Status", interactive=False, elem_classes="status-alive")
            results_output = gr.Markdown(label="Evaluation Report")
            
            news_output_html = gr.HTML()

    evaluate_btn.click(
        fn=run_evaluation,
        inputs=[images_input, student_name_input, subject_input, grading_rubric_input, teacher_key_input, total_marks_input],
        outputs=[status_output, results_output]
    )
    
    demo.load(
        fn=fetch_education_news,
        inputs=None,
        outputs=[news_output_html]
    )

if __name__ == "__main__":
    print("Launching Exam Lense Frontend Premium Dashboard...")
    demo.launch(server_name="127.0.0.1", server_port=8501, favicon_path="assets/favicon.png", css=custom_css, theme=gr.themes.Base())
