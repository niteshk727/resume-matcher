from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi. staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
import io


# Create FastAPI instance
app = FastAPI()

# Load Sentence-BERT model (runs once at startup)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

# Mount the templates folder
templates = Jinja2Templates(directory="templates")

# Serve static files (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
def home():
    return {"message": "Welcome to the Resume Matcher API!"}

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using pdfplumber.
    
    Parameters:
        pdf_file (BytesIO): The uploaded PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

@app.post("/extract_text/")
async def extract_text(resume: UploadFile = File(...)):
    """
    API endpoint to extract text from an uploaded resume PDF.
    
    Parameters:
        resume (UploadFile): The uploaded PDF file.
    
    Returns:
        dict: Extracted text from the PDF.
    """
    pdf_bytes = io.BytesIO(await resume.read())  # Read uploaded file
    extracted_text = extract_text_from_pdf(pdf_bytes)
    
    return {"extracted_text": extracted_text}

def compute_match(job_desc, resume_text):
    """
    Compares job description with resume text and returns match percentage.
    
    Parameters:
        job_desc (str): Job description text.
        resume_text (str): Extracted text from resume.
    
    Returns:
        float: Match percentage (0 to 100).
    """
    # Encode both texts into embeddings
    embeddings = model.encode([job_desc, resume_text], convert_to_tensor=True)
    
    # Compute cosine similarity (range: -1 to 1)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    # Convert to percentage
    return round(similarity_score * 100, 2)

@app.post("/match_resume/")
async def match_resume(job_description: str = Form(...), resume: UploadFile = File(...)):
    """
    API endpoint to compare a job description with a resume and suggest missing skills.

    Returns:
        dict: Match percentage, extracted resume text, and missing keywords.
    """
    pdf_bytes = io.BytesIO(await resume.read())
    resume_text = extract_text_from_pdf(pdf_bytes)
    
    match_percentage = compute_match(job_description, resume_text)
    missing_skills = extract_keywords(job_description, resume_text)

    return {
        "match_percentage": match_percentage,
        "missing_keywords": missing_skills,
        "extracted_text": resume_text
    }

def extract_keywords(job_desc, resume_text, top_n=10):
    """
    Extracts important keywords from job description and finds missing ones in resume.

    Parameters:
        job_desc (str): Job description text.
        resume_text (str): Extracted resume text.
        top_n (int): Number of top keywords to consider.

    Returns:
        list: List of missing keywords.
    """
    # TF-IDF Vectorizer to find important words
    vectorizer = TfidfVectorizer(stop_words="english")
    
    # Fit on job description text
    job_words = vectorizer.fit_transform([job_desc])
    job_keywords = vectorizer.get_feature_names_out()

    # Check which keywords are missing in the resume
    missing_keywords = [word for word in job_keywords if word.lower() not in resume_text.lower()]

    return missing_keywords[:top_n]