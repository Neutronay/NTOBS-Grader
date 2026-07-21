import os
import uuid
import json
import io
import time
import shutil
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, send_file
from werkzeug.utils import secure_filename
from pydantic import BaseModel, Field
from typing import List

# PDF processing libraries
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Google GenAI SDK
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "classmark-secret-key-12345")

# ---- Setup Directories ----
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'graded_outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ---- Setup Gemini Client ----
api_key = os.environ.get("GEMINI_API_KEY")

if os.environ.get("PYTHONANYWHERE_SITE"):
    os.environ["HTTP_PROXY"] = "http://proxy.server:3128"
    os.environ["HTTPS_PROXY"] = "http://proxy.server:3128"

client = genai.Client(api_key=api_key)

# ---- Pydantic Schemas for Structured Output ----
class SubScore(BaseModel):
    question_number: str
    score_awarded: float
    feedback_comment: str

class Annotation(BaseModel):
    page: int = Field(description="0-indexed page number where comment applies")
    x_percent: float = Field(description="X coordinate from left (0 to 100)")
    y_percent: float = Field(description="Y coordinate from top (0 to 100)")
    comment: str

class GradingResult(BaseModel):
    score: float
    breakdown: List[SubScore]
    overall_feedback: str
    annotations: List[Annotation]


# ---- Core Business Logic Routes ----

@app.route('/', methods=['GET', 'POST'])
def index():
    """Step 1: Home Page - Teacher Setup & Context Upload"""
    if request.method == 'POST':
        session['subject'] = request.form.get('subject')
        session['class_section'] = request.form.get('class_section')
        
        try:
            session['num_students'] = int(request.form.get('num_students', 1))
        except ValueError:
            session['num_students'] = 1

        try:
            session['total_score'] = float(request.form.get('total_score', 100))
        except ValueError:
            session['total_score'] = 100.0

        # Handle Rubric/Marking Guide Upload
        rubric_file = request.files.get('rubric_file')
        if rubric_file and rubric_file.filename != '':
            filename = secure_filename(rubric_file.filename)
            rubric_path = os.path.join(UPLOAD_FOLDER, f"rubric_{uuid.uuid4().hex}_{filename}")
            rubric_file.save(rubric_path)
            session['rubric_path'] = rubric_path

            if filename.lower().endswith('.txt'):
                try:
                    with open(rubric_path, 'r', encoding='utf-8') as f:
                        session['rubric_text'] = f.read()
                except Exception as e:
                    print(f"Error reading text rubric file: {e}")
                    session['rubric_text'] = f"Master marking guide document reference: {filename}"
            else:
                session['rubric_text'] = f"Master marking guide document reference: {filename}"

        return redirect(url_for('batch_ingestion'))

    return render_template('index.html')


@app.route('/batch-ingestion', methods=['GET', 'POST'])
def batch_ingestion():
    """Step 2: Batch Roster Configuration and File Matcher"""
    num_students = session.get('num_students', 1)
    parsed_students = []

    if request.method == 'POST' and 'roster_file' in request.files:
        roster_file = request.files['roster_file']
        if roster_file and roster_file.filename != '':
            filename = secure_filename(roster_file.filename)
            path = os.path.join(UPLOAD_FOLDER, f"roster_{uuid.uuid4().hex}_{filename}")
            roster_file.save(path)

            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)

                name_col = [col for col in df.columns if 'name' in str(col).lower()]
                if name_col:
                    parsed_students = df[name_col[0]].dropna().astype(str).tolist()
                    session['num_students'] = len(parsed_students)
            except Exception as e:
                print(f"Error parsing roster file: {e}")
            finally:
                if os.path.exists(path):
                    os.remove(path)

    return render_template(
        'batch.html',
        num_students=session.get('num_students', num_students),
        parsed_students=parsed_students
    )


@app.route('/process-grading', methods=['POST'])
def process_grading():
    """Step 3 & 4: Core Engine Execution Route"""
    student_names = request.form.getlist('student_names[]')
    student_files = request.files.getlist('student_scripts[]')

    rubric_text = session.get('rubric_text', 'Follow standard logical grading rules.')
    total_score = session.get('total_score', 100.0)

    results_summary = []

    for index, name in enumerate(student_names):
        if index >= len(student_files):
            break

        file = student_files[index]
        if not file or file.filename == '':
            continue

        filename = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex
        script_path = os.path.join(UPLOAD_FOLDER, f"{unique_prefix}_{filename}")
        file.save(script_path)

        grading_data = None
        uploaded_media = None

        # 1. AI Grading Processing using google-genai SDK
        try:
            uploaded_media = client.files.upload(
                file=script_path,
                config=types.UploadFileConfig(mime_type="application/pdf")
            )

            while uploaded_media.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_media = client.files.get(name=uploaded_media.name)

            if uploaded_media.state.name == "FAILED":
                raise ValueError("PDF File processing failed on Gemini server.")

            prompt_content = f"""
            You are a highly thorough academic grader. Assess the student script provided.
            Reference Marking Rubric/Guide context: {rubric_text}
            Maximum possible total score: {total_score}

            Carefully cross-verify every written answer against the criteria. Assign individual question points and generate layout mapping annotations for exact visual points on pages.
            """

            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[uploaded_media, prompt_content],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GradingResult,
                    temperature=0.1
                )
            )

            grading_data = json.loads(response.text)

        except Exception as e:
            print(f"Gemini API Exception processing student {name}: {e}")
            grading_data = {
                "score": 0.0,
                "breakdown": [],
                "overall_feedback": f"Failed to evaluate script via AI: {str(e)}",
                "annotations": []
            }

        finally:
            if uploaded_media:
                try:
                    client.files.delete(name=uploaded_media.name)
                except Exception as del_err:
                    print(f"Failed to delete uploaded media {uploaded_media.name}: {del_err}")

        # 2. Document Layout Overlay Engine
        graded_filename = f"Graded_{unique_prefix}_{filename}"
        graded_pdf_path = os.path.join(OUTPUT_FOLDER, graded_filename)
        annotate_student_pdf(script_path, graded_pdf_path, grading_data)

        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except Exception as e:
                print(f"Error removing temporary upload file: {e}")

        results_summary.append({
            "name": name,
            "score": grading_data.get("score", 0),
            "feedback": grading_data.get("overall_feedback", ""),
            "download_url": url_for('download_file', filename=graded_filename)
        })

    session['grading_results'] = results_summary
    return redirect(url_for('results_dashboard'))


def annotate_student_pdf(input_pdf_path, output_pdf_path, grading_data):
    """Draws red circle grades and comment callouts natively onto student's work canvas dimensions."""
    try:
        reader = PdfReader(input_pdf_path)
        writer = PdfWriter()

        annotations = grading_data.get('annotations', [])

        for page_idx, page in enumerate(reader.pages):
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)

            page_annotations = []
            for a in annotations:
                p_num = a.get('page', 0) if isinstance(a, dict) else getattr(a, 'page', 0)
                if int(p_num) == page_idx:
                    page_annotations.append(a)

            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))

            if page_idx == 0:
                can.setFillColorRGB(0.85, 0.1, 0.1)
                can.setStrokeColorRGB(0.85, 0.1, 0.1)
                can.setLineWidth(3)
                can.circle(page_width - 70, page_height - 70, 35, stroke=1, fill=0)
                can.setFont("Helvetica-Bold", 16)
                score_val = grading_data.get('score', 0)
                can.drawCentredString(page_width - 70, page_height - 76, f"{score_val}")

            for ann in page_annotations:
                x_val = ann.get('x_percent', 50) if isinstance(ann, dict) else getattr(ann, 'x_percent', 50)
                y_val = ann.get('y_percent', 50) if isinstance(ann, dict) else getattr(ann, 'y_percent', 50)
                comment_val = ann.get('comment', '') if isinstance(ann, dict) else getattr(ann, 'comment', '')

                x_pct = float(x_val) / 100.0
                y_pct = float(y_val) / 100.0

                target_x = x_pct * page_width
                target_y = page_height - (y_pct * page_height)

                can.setFillColorRGB(0.85, 0.1, 0.1)
                can.setStrokeColorRGB(0.85, 0.1, 0.1)
                can.setLineWidth(1.5)

                can.circle(target_x, target_y, 8, stroke=1, fill=0)
                can.setFont("Helvetica-Bold", 9)
                can.drawString(target_x + 12, target_y - 3, str(comment_val))

            can.save()
            packet.seek(0)

            overlay_reader = PdfReader(packet)
            if len(overlay_reader.pages) > 0:
                page.merge_page(overlay_reader.pages[0])

            writer.add_page(page)

        with open(output_pdf_path, 'wb') as f:
            writer.write(f)

    except Exception as e:
        print(f"Error annotating PDF document target structure: {e}")
        if os.path.exists(input_pdf_path):
            shutil.copy(input_pdf_path, output_pdf_path)


@app.route('/results')
def results_dashboard():
    """Step 4: Interactive Dashboard Rendering Output Records Data"""
    results = session.get('grading_results', [])
    return render_template('results.html', results=results)


@app.route('/download/<path:filename>')
def download_file(filename):
    """File download handler for graded PDFs"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/export/xlsx')
def export_xlsx():
    """Generates standard XLSX Gradebook Sheet matrix summary natively."""
    results = session.get('grading_results', [])
    if not results:
        return "No data to export", 400

    df = pd.DataFrame(results)[['name', 'score', 'feedback']]
    df.columns = ['Student Name', 'Score Awarded', 'Overall Feedback Evaluated']

    out_io = io.BytesIO()
    with pd.ExcelWriter(out_io, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ClassMark Summary')

    out_io.seek(0)
    return send_file(
        out_io,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="ClassMark_AI_Gradebook.xlsx"
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)