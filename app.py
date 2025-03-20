from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tempfile
import langid
from PyPDF2 import PdfReader
import docx
import re
import spacy
from dotenv import load_dotenv
import json
import traceback
import logging
from flask_cors import CORS

from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

try:
    project_id = "medflow-poc-453615"
    location = "us-central1"
    logger.info(f"Initializing Vertex AI with project: {project_id}, location: {location}")
    vertexai.init(project=project_id, location=location)
    
    import google.auth
    credentials, project = google.auth.default()
    logger.info(f"Using Google Cloud project: {project}")
except Exception as e:
    logger.error(f"Error initializing Vertex AI: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Loading English NLP model")
    nlp_en = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Error loading English model: {str(e)}, attempting to download")
    try:
        os.system("python -m spacy download en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded and loaded English model")
    except Exception as e:
        logger.error(f"Failed to download English model: {str(e)}")

try:
    logger.info("Loading French NLP model")
    nlp_fr = spacy.load("fr_core_news_sm")
except Exception as e:
    logger.warning(f"Error loading French model: {str(e)}, attempting to download")
    try:
        os.system("python -m spacy download fr_core_news_sm")
        nlp_fr = spacy.load("fr_core_news_sm")
        logger.info("Successfully downloaded and loaded French model")
    except Exception as e:
        logger.error(f"Failed to download French model: {str(e)}")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file with error handling"""
    try:
        logger.info(f"Extracting text from PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.debug(f"Extracted {len(page_text)} characters from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {str(e)}")
        
        logger.info(f"Extracted total of {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_text_from_docx(file_path):
    """Extract text from DOCX file with error handling"""
    try:
        logger.info(f"Extracting text from DOCX: {file_path}")
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        text = '\n'.join(full_text)
        logger.info(f"Extracted total of {len(text)} characters from DOCX")
        return text
    except Exception as e:
        logger.error(f"Error in extract_text_from_docx: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def detect_language(text):
    """Detect language with error handling"""
    try:
        logger.info("Detecting language")
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for language detection, defaulting to English")
            return "en"
        
        lang, confidence = langid.classify(text)
        logger.info(f"Detected language: {lang} with confidence: {confidence}")
        return lang
    except Exception as e:
        logger.error(f"Error in detect_language: {str(e)}")
        logger.error(traceback.format_exc())
        return "en"

def extract_skills(text, language):
    """Extract skills from text with error handling"""
    try:
        logger.info(f"Extracting skills from text (language: {language})")
        
        if language not in ["en", "fr"]:
            logger.warning(f"Unsupported language: {language}, defaulting to English")
            language = "en"
        
        if language == "en":
            nlp = nlp_en
            skill_keywords = [
                "python", "javascript", "java", "c\\+\\+", "c#", "react", "node", "angular",
                "vue", "html", "css", "sql", "nosql", "mongodb", "docker", "kubernetes",
                "aws", "azure", "gcp", "cloud", "git", "machine learning", "deep learning",
                "ai", "data science", "analytics", "leadership", "management", "communication",
                "teamwork", "problem solving", "critical thinking", "project management",
                "agile", "scrum", "product management", "ux", "ui", "design", "research",
                "marketing", "sales", "customer service", "engineering", "devops", "security"
            ]
        else:
            nlp = nlp_fr
            skill_keywords = [
                "python", "javascript", "java", "c\\+\\+", "c#", "react", "node", "angular",
                "vue", "html", "css", "sql", "nosql", "mongodb", "docker", "kubernetes",
                "aws", "azure", "gcp", "cloud", "git", "apprentissage automatique", "apprentissage profond",
                "ia", "science des données", "analytique", "leadership", "gestion", "communication",
                "travail d'équipe", "résolution de problèmes", "pensée critique", "gestion de projet",
                "agile", "scrum", "gestion de produit", "ux", "ui", "conception", "recherche",
                "marketing", "ventes", "service client", "ingénierie", "devops", "sécurité"
            ]
        
        pattern = r'\b(?:' + '|'.join(skill_keywords) + r')\b'
        skills_found = set(re.findall(pattern, text.lower()))
        logger.debug(f"Found {len(skills_found)} skills via regex")
        
        text_for_spacy = text[:50000] if len(text) > 50000 else text
        
        try:
            doc = nlp(text_for_spacy)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"]:
                    skills_found.add(ent.text.lower())
            logger.debug(f"Found {len(skills_found)} skills after NER")
        except Exception as e:
            logger.error(f"Error during NER processing: {str(e)}")
        
        return list(skills_found)
    except Exception as e:
        logger.error(f"Error in extract_skills: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def generate_feedback(text, language):
    """Generate feedback using Vertex AI with error handling"""
    try:
        logger.info("Generating feedback with Vertex AI")
        
        max_length = 10000
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            text = text[:max_length]
        
        model = GenerativeModel("gemini-pro")
        logger.debug("Initialized Gemini Pro model")
        
        if language == "en":
            prompt = f"""
            You are a professional resume reviewer. Analyze the following resume and provide:
            
            1. Constructive feedback on how to improve the resume (focus on structure, content, clarity, and impact)
            2. 5 recommended certifications with their official website URLs and short descriptions that would complement the skills and experience
            3. Suggested career paths based on the resume
            
            Resume text:
            {text}
            
            Format your response as a JSON object with the following structure:
            {{
              "feedback": "detailed feedback here",
              "certifications": [
                {{
                  "name": "certification name",
                  "url": "official website URL",
                  "description": "short description"
                }}
              ],
              "careerPath": "detailed career path suggestions here"
            }}
            
            IMPORTANT: Your response MUST be valid JSON. Do not include any additional text or Markdown formatting.
            """
        else:
            prompt = f"""
            Vous êtes un évaluateur professionnel de CV. Analysez le CV suivant et fournissez:
            
            1. Des commentaires constructifs sur la façon d'améliorer le CV (concentrez-vous sur la structure, le contenu, la clarté et l'impact)
            2. 5 certifications recommandées avec leurs URLs officielles et de courtes descriptions qui compléteraient les compétences et l'expérience
            3. Des suggestions de parcours professionnels basés sur le CV
            
            Texte du CV:
            {text}
            
            Formatez votre réponse sous forme d'objet JSON avec la structure suivante:
            {{
              "feedback": "commentaires détaillés ici",
              "certifications": [
                {{
                  "name": "nom de la certification",
                  "url": "URL officielle",
                  "description": "courte description"
                }}
              ],
              "careerPath": "suggestions détaillées de parcours professionnel ici"
            }}
            
            IMPORTANT: Votre réponse DOIT être un JSON valide. N'incluez aucun texte supplémentaire ou formatage Markdown.
            """
        
        logger.debug("Sending prompt to Gemini Pro")
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini Pro")
        
        response_text = response.text
        logger.debug(f"Raw response preview: {response_text[:200]}...")
        
        return response_text
    except Exception as e:
        logger.error(f"Error in generate_feedback: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    logger.info("Test endpoint called")
    
    try:
        import google.auth
        credentials, project = google.auth.default()
        cloud_status = f"Google Cloud authenticated with project: {project}"
        logger.info(cloud_status)
    except Exception as e:
        cloud_status = f"Google Cloud authentication error: {str(e)}"
        logger.error(cloud_status)
    
    return jsonify({
        "status": "ok",
        "message": "API is working",
        "google_cloud": cloud_status
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main endpoint to analyze resumes"""
    try:
        logger.info("Analyze resume endpoint called")
        
        if 'resume' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['resume']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)
        logger.info(f"File saved to {file_path}")
        
        try:
            if file.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.filename.lower().endswith(('.doc', '.docx')):
                text = extract_text_from_docx(file_path)
            else:
                os.remove(file_path)
                logger.error(f"Unsupported file format: {file.filename}")
                return jsonify({"error": "Unsupported file format"}), 400
            
            logger.info(f"Text extracted, length: {len(text)}")
            
            if not text or len(text.strip()) == 0:
                logger.error("No text could be extracted from the file")
                return jsonify({"error": "No text could be extracted from the file"}), 400
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            os.remove(file_path)
            return jsonify({"error": f"Error extracting text: {str(e)}"}), 500
        
        language = detect_language(text)
        logger.info(f"Detected language: {language}")
        
        skills = extract_skills(text, language)
        logger.info(f"Extracted skills: {skills}")
        
        try:
            ai_feedback_text = generate_feedback(text, language)
            logger.info("Successfully generated AI feedback")
            
            ai_feedback_text = ai_feedback_text.strip()
            
            if ai_feedback_text.startswith("```json") and ai_feedback_text.endswith("```"):
                ai_feedback_text = ai_feedback_text[7:-3].strip()
            elif ai_feedback_text.startswith("```") and ai_feedback_text.endswith("```"):
                ai_feedback_text = ai_feedback_text[3:-3].strip()
            
            try:
                ai_feedback = json.loads(ai_feedback_text)
                logger.info("Successfully parsed AI feedback as JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse response as JSON: {str(e)}")
                
                try:
                    json_start = ai_feedback_text.find('{')
                    json_end = ai_feedback_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = ai_feedback_text[json_start:json_end]
                        ai_feedback = json.loads(json_str)
                        logger.info("Successfully extracted and parsed JSON object")
                    else:
                        raise ValueError("No JSON object found in response")
                except Exception as e:
                    logger.error(f"Failed to extract JSON from response: {str(e)}")
                    logger.error(f"Raw response: {ai_feedback_text}")
                    ai_feedback = {
                        "feedback": "An error occurred while parsing the AI response. Please try again.",
                        "certifications": [],
                        "careerPath": "Unable to generate career path suggestions."
                    }
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            ai_feedback = {
                "feedback": f"An error occurred while generating feedback: {str(e)}",
                "certifications": [],
                "careerPath": "Unable to generate career path suggestions."
            }
        
        try:
            os.remove(file_path)
            logger.info("Temp file removed")
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {str(e)}")
        
        return jsonify({
            "skills": skills,
            "feedback": ai_feedback.get("feedback", "No feedback available"),
            "certifications": ai_feedback.get("certifications", []),
            "careerPath": ai_feedback.get("careerPath", "No career path suggestions available"),
            "language": language
        })
    except Exception as e:
        logger.error(f"Unexpected error in analyze_resume: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5000)