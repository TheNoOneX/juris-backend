"""
Juris-AI Backend – main.py (OPTIMIZED)

"""

# ======================================================
# IMPORTS
# ======================================================

import os
import io
import re
import json
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import asyncio

import pytesseract
import pdfplumber
from docx import Document
from google import genai


# ======================================================
# CONFIGURATION CONSTANTS
# ======================================================

# Text Processing
MAX_TEXT_LENGTH = 10000  # Maximum characters to send to AI
MAX_FILE_SIZE_MB = 10  # Maximum file upload size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Risk Scoring Thresholds
RISK_HIGH_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 40

# API Configuration
API_TIMEOUT_SECONDS = 60  # Timeout for Gemini API calls
MAX_REFERENCES = 5  # Maximum number of references to include

# Supported Languages
SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu"]

# Supported Verbosity Levels
SUPPORTED_VERBOSITY = ["basic", "medium", "advanced"]


# ======================================================
# ENV & APP INITIALIZATION
# ======================================================

print("🔧 Loading environment variables...")
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY missing in .env file")

print("🔌 Connecting to Gemini...")
try:
    client = genai.Client(api_key=API_KEY)
    MODEL_NAME = "gemini-2.5-flash"
    print("✅ Gemini connected successfully")
except Exception as e:
    print(f"❌ Failed to connect to Gemini: {e}")
    raise

app = FastAPI(
    title="JURIS-AI Backend",
    version="6.0",
    description="Indian Legal Risk Analysis API"
)

# ======================================================
# CORS CONFIGURATION (FIXED)
# ======================================================

# Support multiple origins for development and production
ALLOWED_ORIGINS = [
    "https://juris-ai-pro.netlify.app",  # Production (no trailing slash!)
    "http://localhost:3000",  # Local development
    "http://localhost:8000",  # Local API testing
    "http://127.0.0.1:3000",  # Alternative localhost
    "http://127.0.0.1:8000",  # Alternative localhost
    "http://localhost:5500",  # Live Server default port
    "http://127.0.0.1:5500",  # Live Server alternative
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 FastAPI initialized with CORS support")


# ======================================================
# SYSTEM PROMPT (ENHANCED)
# ======================================================

SYSTEM_PROMPT = """
You are JURIS AI, an Indian legal risk analysis assistant.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
1. Output ONLY valid JSON - no markdown, no preamble, no explanation.
2. ALL fields MUST exist in the response.
3. ALL list fields MUST be FLAT ARRAYS OF STRINGS.
4. DO NOT return objects, dictionaries, or nested structures in lists.
5. DO NOT group items or use categories.
6. Verbosity affects DEPTH and DETAIL ONLY, never structure.
7. Never hallucinate laws or cases - only cite real Indian laws.
8. If uncertain, mark risk as MEDIUM and explain limitations.

Risk Scoring Rules:
- HIGH: score > 70 (Serious legal issues, immediate action required)
- MEDIUM: score 40–70 (Potential concerns, review recommended)
- LOW: score < 40 (Minor or no significant legal risks)

Response must be valid JSON matching this exact schema:
{
  "summary": "string",
  "red_flags": ["string array"],
  "laws": ["string array"],
  "actions": ["string array"],
  "final_summary": ["string array"],
  "references": ["string array"],
  "risk_score": number (0-100),
  "risk_level": "HIGH|MEDIUM|LOW"
}
"""


# ======================================================
# VERBOSITY CONTEXT (IMPROVED)
# ======================================================

PROMPT_STYLE = {
    "basic": """
VERBOSITY MODE: BASIC
- Use very simple, everyday language
- Try to explain legal terms if exist in document in very simple language
- Assume the user has no legal knowledge
- Keep explanations to 1-2 short sentences per point
- Do NOT use section numbers or legal jargon
- Focus only on practical impact to the user
- Explain what it means in real life
""",
    "medium": """
VERBOSITY MODE: MEDIUM
- Use clear, informative language
- Explain WHY something is risky or important
- Mention law names (e.g., "Indian Contract Act") but avoid section numbers
- Explain consequences in understandable terms
- Balance legal accuracy with accessibility
- Provide context for recommendations
""",
    "advanced": """
VERBOSITY MODE: ADVANCED
- Use precise legal terminology
- Provide detailed legal reasoning and analysis
- Cite specific Acts AND Section numbers (e.g., "Section 420 of IPC")
- Explain legal principles, precedents, and enforceability
- Include procedural details where relevant
- Do NOT dump unnecessary case law - cite only when critical
- Maintain professional legal writing standards
"""
}

def get_style(verbosity: str) -> str:
    """
    Returns the appropriate verbosity prompt based on user selection.
    Falls back to 'basic' if invalid verbosity level provided.
    """
    return PROMPT_STYLE.get(verbosity, PROMPT_STYLE["basic"])


# ======================================================
# LANGUAGE RULE
# ======================================================

def language_rule(language: str) -> str:
    """
    Generates language instruction for the AI.
    Ensures all responses are in the user's selected language.
    """
    return f"""
LANGUAGE RULE:
- Respond ONLY in {language}
- Translate all legal terms into {language} with brief explanations
- Maintain natural, fluent {language} throughout
- Do NOT mix languages
"""


# ======================================================
# INPUT VALIDATION
# ======================================================

def validate_language(language: str) -> str:
    """
    Validates and sanitizes language input.
    Returns validated language or defaults to English.
    """
    if language in SUPPORTED_LANGUAGES:
        return language
    print(f"⚠️ Invalid language '{language}', defaulting to English")
    return "English"


def validate_verbosity(verbosity: str) -> str:
    """
    Validates and sanitizes verbosity input.
    Returns validated verbosity or defaults to basic.
    """
    if verbosity in SUPPORTED_VERBOSITY:
        return verbosity
    print(f"⚠️ Invalid verbosity '{verbosity}', defaulting to basic")
    return "basic"


def validate_text_input(text: str) -> str:
    """
    Validates and sanitizes text input.
    Removes excessive whitespace and limits length.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Limit length
    if len(text) > MAX_TEXT_LENGTH:
        print(f"⚠️ Text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters")
        return text[:MAX_TEXT_LENGTH]
    
    return text


# ======================================================
# DOCUMENT CLASSIFICATION
# ======================================================

def classify_document(text: str) -> str:
    """
    Classifies document type based on content analysis.
    Helps AI provide more contextually relevant analysis.
    """
    text_lower = text.lower()

    # Check for university/academic documents
    if any(keyword in text_lower for keyword in [
        "exam", "university", "semester", "controller of examinations",
        "academic", "transcript", "degree"
    ]):
        return "university_circular"
    
    # Check for court/legal notices
    if any(keyword in text_lower for keyword in [
        "legal notice", "summons", "advocate", "high court", "supreme court",
        "petition", "plaintiff", "defendant"
    ]):
        return "court_notice"
    
    # Check for government notices
    if any(keyword in text_lower for keyword in [
        "gazette", "ministry", "government of india", "notification",
        "ordinance", "circular"
    ]):
        return "government_notice"
    
    # Check for insurance documents
    if any(keyword in text_lower for keyword in [
        "insurance", "policy number", "sum insured", "premium",
        "policyholder", "beneficiary"
    ]):
        return "insurance_policy"

    # Default to general agreement/contract
    return "agreement"


# ======================================================
# PROMPT BUILDER
# ======================================================

def build_prompt(
    text: str,
    doc_type: str,
    language: str,
    verbosity: str
) -> str:
    """
    Constructs the complete prompt for AI analysis.
    Combines system prompt, language rules, verbosity settings, and content.
    """
    return f"""
{SYSTEM_PROMPT}

{language_rule(language)}

{get_style(verbosity)}

DOCUMENT TYPE: {doc_type}

TASK:
Analyze the document below and populate ALL fields according to the rules.
Ensure all arrays contain strings, not objects.
Be accurate - do not hallucinate laws or legal provisions.

DOCUMENT TEXT:
\"\"\"{text[:MAX_TEXT_LENGTH]}\"\"\"

OUTPUT FORMAT (JSON ONLY):
{{
  "summary": "",
  "red_flags": [],
  "laws": [],
  "actions": [],
  "final_summary": [],
  "references": [],
  "risk_score": 0,
  "risk_level": ""
}}
"""


# ======================================================
# GEMINI API CALL WITH TIMEOUT
# ======================================================

async def call_gemini_with_timeout(prompt: str, file_part=None) -> Any:
    """
    Calls Gemini API with timeout protection.
    Prevents hanging requests and provides better error handling.
    """
    try:
        # Create the API call as a coroutine
        parts = [{"text": prompt}]
        if file_part:
            parts.append(file_part)
        
        # Run with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=[{"role": "user", "parts": parts}]
            ),
            timeout=API_TIMEOUT_SECONDS
        )
        
        return response
        
    except asyncio.TimeoutError:
        print(f"❌ Gemini API timeout after {API_TIMEOUT_SECONDS} seconds")
        raise HTTPException(
            status_code=504,
            detail="AI analysis timed out. Please try with a shorter document."
        )
    except Exception as e:
        print(f"❌ Gemini API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI service error: {str(e)}"
        )


# ======================================================
# JSON EXTRACTION (IMPROVED)
# ======================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts JSON from AI response with better error handling.
    Handles markdown code blocks and malformed responses.
    """
    try:
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            raise ValueError("No JSON object found in response")
        
        json_str = match.group()
        parsed = json.loads(json_str)
        
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error at position {e.pos}: {e.msg}")
        print(f"Response preview: {text[:200]}...")
        return {}
    except Exception as e:
        print(f"❌ Unexpected error extracting JSON: {e}")
        return {}


# ======================================================
# SCHEMA ENFORCEMENT (OPTIMIZED)
# ======================================================

def enforce_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces consistent response schema with type safety.
    Ensures all required fields exist with correct types.
    Optimized with early returns and reduced checks.
    """
    
    def flatten_to_string_list(value: Any) -> list[str]:
        """
        Converts any value to a flat list of strings.
        Handles nested structures and various input types.
        """
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    # If dict, extract values
                    result.extend(str(v) for v in item.values() if v)
                elif isinstance(item, str):
                    result.append(item)
                else:
                    result.append(str(item))
            return result
        
        if isinstance(value, dict):
            return [str(v) for v in value.values() if v]
        
        if isinstance(value, str):
            return [value] if value.strip() else []
        
        return []

    # Extract and validate risk score
    risk_score = data.get("risk_score", 50)
    try:
        risk_score = int(risk_score)
        # Clamp between 0-100
        risk_score = max(0, min(100, risk_score))
    except (ValueError, TypeError):
        print("⚠️ Invalid risk_score, defaulting to 50")
        risk_score = 50

    # Determine risk level based on score
    if risk_score > RISK_HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif risk_score >= RISK_MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Process references with fallback
    references = flatten_to_string_list(data.get("references", []))
    
    # Add default references if list is too short
    if len(references) < 2:
        default_refs = [
            "https://www.indiacode.nic.in - India Code (Central Acts)",
            "https://main.sci.gov.in - Supreme Court of India"
        ]
        references.extend(default_refs)
    
    # Limit to max references
    references = references[:MAX_REFERENCES]

    # Build final schema
    return {
        "summary": str(data.get("summary", "No summary available")).strip(),
        "red_flags": flatten_to_string_list(data.get("red_flags", [])),
        "laws": flatten_to_string_list(data.get("laws", [])),
        "actions": flatten_to_string_list(data.get("actions", [])),
        "final_summary": flatten_to_string_list(data.get("final_summary", [])),
        "references": references,
        "risk_score": risk_score,
        "risk_level": risk_level
    }


# ======================================================
# FILE PROCESSING UTILITIES
# ======================================================

def make_file_part(file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Converts raw file bytes into a Gemini-compatible inline_data part.
    Used for images, scanned PDFs, or any binary document.
    """
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(file_bytes).decode("utf-8")
        }
    }


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extracts text from image using OCR (Tesseract).
    Returns empty string if OCR fails or no text found.
    """
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF using pdfplumber.
    Handles multi-page PDFs efficiently.
    Returns empty string if extraction fails.
    """
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return ""


def extract_text_from_doc(file_bytes: bytes) -> str:
    """
    Extracts text from Word documents (.doc, .docx).
    Returns empty string if extraction fails.
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"❌ Word document extraction error: {e}")
        return ""


# ======================================================
# ERROR HANDLER
# ======================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for better error responses.
    Provides structured error messages to frontend.
    """
    print(f"❌ Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "detail": str(exc) if os.getenv("DEBUG") else None
        }
    )


# ======================================================
# API ROUTES
# ======================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    Returns API status and version.
    """
    return {
        "status": "online",
        "version": "6.0",
        "service": "JURIS-AI Legal Analysis API"
    }


@app.get("/health")
def health_check():
    """
    Detailed health check with service status.
    """
    return {
        "status": "healthy",
        "gemini": "connected",
        "model": MODEL_NAME,
        "max_file_size_mb": MAX_FILE_SIZE_MB
    }


@app.post("/api/analyze-text")
async def analyze_text(
    problem: str = Form(...),
    language: str = Form("English"),
    verbosity: str = Form("basic")  # ✅ FIXED: Now accepts verbosity parameter
):
    """
    Analyzes user-provided text for legal risks.
    
    Parameters:
    - problem: The legal problem/text to analyze
    - language: Output language (English, Hindi, Marathi, Tamil, Telugu)
    - verbosity: Detail level (basic, medium, advanced)
    
    Returns:
    - JSON with legal analysis including risk score, recommendations, etc.
    """
    print(f"📨 Text analysis | lang={language} | verbosity={verbosity}")
    
    try:
        # Validate inputs
        language = validate_language(language)
        verbosity = validate_verbosity(verbosity)
        text = validate_text_input(problem)
        
        # Build prompt and call AI
        prompt = build_prompt(text, "agreement", language, verbosity)
        response = await call_gemini_with_timeout(prompt)
        
        # Extract and enforce schema
        result = extract_json(response.text)
        final_result = enforce_schema(result)
        
        print(f"✅ Analysis complete | risk={final_result['risk_level']} | score={final_result['risk_score']}")
        return final_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in analyze_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    language: str = Form("English"),
    verbosity: str = Form("basic")  # ✅ FIXED: Now accepts verbosity parameter
):
    """
    Analyzes uploaded document (image, PDF, Word) for legal risks.
    
    Parameters:
    - file: Uploaded document file
    - language: Output language (English, Hindi, Marathi, Tamil, Telugu)
    - verbosity: Detail level (basic, medium, advanced)
    
    Returns:
    - JSON with legal analysis including risk score, recommendations, etc.
    """
    print(f"📨 Document received: {file.filename} | lang={language} | verbosity={verbosity}")
    
    try:
        # Validate inputs
        language = validate_language(language)
        verbosity = validate_verbosity(verbosity)
        
        # Read file with size limit
        file_bytes = await file.read()
        
        # Check file size
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        
        text = ""
        
        # Process based on file type
        if file.content_type and file.content_type.startswith("image/"):
            # Image file - try OCR first
            print("📸 Processing image file...")
            image = Image.open(io.BytesIO(file_bytes))
            text = extract_text_from_image(image)
            
            if not text:
                # OCR failed - use Gemini Vision
                print("📸 OCR empty → Using Gemini Vision")
                file_part = make_file_part(file_bytes, file.content_type)
                prompt = build_prompt(
                    "Analyze the image content and extract all text.",
                    "agreement",
                    language,
                    verbosity
                )
                response = await call_gemini_with_timeout(prompt, file_part)
                result = extract_json(response.text)
                return enforce_schema(result)
        
        elif file.content_type == "application/pdf":
            # PDF file
            print("📄 Processing PDF file...")
            text = extract_text_from_pdf(file_bytes)
        
        elif file.content_type in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            # Word document
            print("📝 Processing Word document...")
            text = extract_text_from_doc(file_bytes)
        
        else:
            # Unknown file type
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # If text extraction failed, use Gemini Vision fallback
        if not text or len(text.strip()) < 10:
            print("📄 Text extraction failed → Using Gemini Vision fallback")
            file_part = make_file_part(file_bytes, file.content_type)
            prompt = build_prompt(
                "Analyze the uploaded document directly and extract all relevant information.",
                "unknown",
                language,
                verbosity
            )
            response = await call_gemini_with_timeout(prompt, file_part)
            result = extract_json(response.text)
            return enforce_schema(result)
        
        # Validate extracted text
        text = validate_text_input(text)
        
        # Classify document type
        doc_type = classify_document(text)
        print(f"📄 Classified as: {doc_type}")
        
        # Build prompt and analyze
        prompt = build_prompt(text, doc_type, language, verbosity)
        response = await call_gemini_with_timeout(prompt)
        
        # Extract and enforce schema
        result = extract_json(response.text)
        final_result = enforce_schema(result)
        
        print(f"✅ Analysis complete | risk={final_result['risk_level']} | score={final_result['risk_score']}")
        return final_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in analyze_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# STARTUP EVENT
# ======================================================

@app.on_event("startup")
async def startup_event():
    """
    Runs on server startup.
    Performs initialization checks.
    """
    print("=" * 50)
    print("🚀 JURIS-AI Backend Starting...")
    print(f"📦 Model: {MODEL_NAME}")
    print(f"🌐 CORS: {len(ALLOWED_ORIGINS)} origins allowed")
    print(f"📁 Max file size: {MAX_FILE_SIZE_MB}MB")
    print(f"⏱️  API timeout: {API_TIMEOUT_SECONDS}s")
    print("=" * 50)


# ======================================================
# RUN SERVER (for development)
# ======================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
