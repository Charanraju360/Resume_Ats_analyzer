from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import PyPDF2
import os
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# IMPORTANT: Define the ImprovedTextPreprocessor class BEFORE loading models
# This class must match exactly the one used during training
# ============================================================================

class ImprovedTextPreprocessor:
    """
    Enhanced text preprocessing for resume and job description text
    """
    def __init__(self):
        # Use a smaller set of stopwords to preserve technical context
        self.stop_words = set(stopwords.words('english')) - {
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
            'over', 'under', 'again', 'further', 'then', 'once'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text while preserving technical terms"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep alphanumeric, spaces, and important symbols like +, #, .
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        """Tokenize and remove only stopwords, NO stemming"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove only stopwords, keep all meaningful terms
        tokens = [token for token in tokens 
                  if token not in self.stop_words and len(token) > 1]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_filter(text)
        return text

# Load the trained model and components
print("Loading model and components...")
try:
    model = joblib.load('ats_model_improved.pkl')
    vectorizer = joblib.load('tfidf_vectorizer_improved.pkl')
    label_encoder = joblib.load('label_encoder_improved.pkl')
    preprocessor = joblib.load('text_preprocessor_improved.pkl')
    
    # Load the original dataset for role information
    df = pd.read_csv('ats_claude.csv')
    
    print("✓ Model and components loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else None
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def extract_skills_from_text(text, role_skills):
    """Extract skills from resume text with improved matching"""
    text_lower = text.lower()
    found_skills = []
    
    skill_variations = {
        'machine learning': ['ml', 'machine learning', 'machine-learning'],
        'artificial intelligence': ['ai', 'artificial intelligence'],
        'deep learning': ['dl', 'deep learning', 'deep-learning'],
        'natural language processing': ['nlp', 'natural language processing'],
        'computer vision': ['cv', 'computer vision'],
        'python': ['python', 'python3', 'python2'],
        'javascript': ['javascript', 'js', 'ecmascript'],
        'typescript': ['typescript', 'ts'],
        'c++': ['c++', 'cpp', 'cplusplus'],
        'c#': ['c#', 'csharp', 'c-sharp'],
    }
    
    for skill in role_skills:
        skill_lower = skill.strip().lower()
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            found_skills.append(skill.strip())
            continue
        
        for standard_skill, variations in skill_variations.items():
            if skill_lower in variations:
                for variant in variations:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    if re.search(pattern, text_lower):
                        found_skills.append(skill.strip())
                        break
    
    return list(set(found_skills))

def calculate_ats_score(resume_text, target_role):
    """Calculate ATS score for the resume"""
    # Preprocess resume text
    processed_text = preprocessor.preprocess(resume_text)
    
    if not processed_text.strip():
        return None
    
    # Vectorize
    text_tfidf = vectorizer.transform([processed_text])
    
    # Get predictions and probabilities
    prediction_proba = model.predict_proba(text_tfidf)[0]
    predicted_role_idx = model.predict(text_tfidf)[0]
    predicted_role = label_encoder.inverse_transform([predicted_role_idx])[0]
    
    # Get probability for target role
    try:
        target_role_idx = np.where(label_encoder.classes_ == target_role)[0][0]
        target_role_probability = prediction_proba[target_role_idx]
    except IndexError:
        return None
    
    # Get required skills for target role
    role_data = df[df['role'] == target_role]
    if role_data.empty:
        return None
    
    role_data = role_data.iloc[0]
    required_skills = [skill.strip() for skill in role_data['skills'].split(',')]
    
    # Extract skills from resume
    found_skills = extract_skills_from_text(resume_text, required_skills)
    missing_skills = [skill for skill in required_skills if skill not in found_skills]
    
    # Calculate skill match percentage
    skill_match_percentage = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
    
    # Calculate text similarity
    role_desc = f"{role_data['skills']}. {role_data['experience_description']}"
    role_processed = preprocessor.preprocess(role_desc)
    role_tfidf = vectorizer.transform([role_processed])
    similarity_score = cosine_similarity(text_tfidf, role_tfidf)[0][0]
    
    # Calculate ATS score (40% confidence + 40% skills + 20% similarity)
    ats_score = (
        target_role_probability * 0.40 +
        (skill_match_percentage / 100) * 0.40 +
        similarity_score * 0.20
    ) * 100
    
    # Get top 3 predicted roles
    top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
    top_3_roles = [
        {
            'role': label_encoder.inverse_transform([idx])[0],
            'probability': round(prediction_proba[idx] * 100, 2)
        }
        for idx in top_3_indices
    ]
    
    return {
        'target_role': target_role,
        'predicted_role': predicted_role,
        'ats_score': round(ats_score, 2),
        'model_confidence': round(target_role_probability * 100, 2),
        'skill_match_percentage': round(skill_match_percentage, 2),
        'text_similarity': round(similarity_score * 100, 2),
        'required_skills': required_skills,
        'found_skills': found_skills,
        'missing_skills': missing_skills,
        'total_required': len(required_skills),
        'total_found': len(found_skills),
        'total_missing': len(missing_skills),
        'top_3_predictions': top_3_roles
    }

@app.route('/')
def index():
    """Render the home page"""
    # Get available roles from the dataset
    available_roles = sorted(df['role'].unique().tolist())
    return render_template('index.html', roles=available_roles)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the uploaded resume"""
    # Check if file was uploaded
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    target_role = request.form.get('role')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not target_role:
        return jsonify({'error': 'No role selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from PDF
            resume_text = extract_text_from_pdf(filepath)
            
            if not resume_text:
                os.remove(filepath)
                return jsonify({'error': 'Could not extract text from PDF'}), 400
            
            # Calculate ATS score
            result = calculate_ats_score(resume_text, target_role)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result is None:
                return jsonify({'error': 'Error analyzing resume'}), 500
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing resume: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400

@app.route('/api/roles', methods=['GET'])
def get_roles():
    """API endpoint to get available roles"""
    available_roles = sorted(df['role'].unique().tolist())
    return jsonify({'roles': available_roles})

if __name__ == '__main__':
    if model is None:
        print("❌ Cannot start server - model failed to load")
    else:
        print("\n" + "="*80)
        print("🚀 ATS RESUME CLASSIFIER SERVER STARTING")
        print("="*80)
        print("📍 Server running at: http://127.0.0.1:5000")
        print("📁 Upload folder: uploads/")
        print("📊 Model loaded: ✓")
        print(f"🎯 Available roles: {len(df['role'].unique())}")
        print("="*80 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)