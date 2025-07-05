import os
import re
import openai
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Clean text for TF-IDF
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

# Extract top N keywords
def extract_keywords(text, top_n=15):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    keywords = sorted(zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0]), key=lambda x: -x[1])
    return [word for word, score in keywords[:top_n]]

# Extract resume content from PDF
def extract_resume_text(resume_path):
    return extract_text(resume_path)

# Read job description from file
def get_job_description(path='job_description.txt'):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

# Generate suggestions using OpenAI GPT model
def generate_ai_suggestions(resume_text, job_description, missing_skills=None):
    prompt = f"""
You are an expert resume advisor.

Job Description:
{job_description}

Resume:
{resume_text}

"""
    if missing_skills:
        prompt += f"Missing Skills: {', '.join(missing_skills)}\n"

    prompt += "Please provide a detailed evaluation of how well the resume matches the job, and suggest specific improvements or missing skills to include."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4o" if available
            messages=[
                {"role": "system", "content": "You are a professional resume reviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI API error: {str(e)}"

# Main resume ranking logic
def rank_resumes(uploaded_resume, resume_folder='resumes', job_description_path='job_description.txt'):
    job_description = get_job_description(job_description_path)
    jd_clean = clean_text(job_description)
    jd_keywords = set(extract_keywords(jd_clean))

    results = []

    resume_path = os.path.join(resume_folder, uploaded_resume)
    resume_text = extract_resume_text(resume_path)
    resume_clean = clean_text(resume_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]

    resume_keywords = set(extract_keywords(resume_clean))
    missing_skills = list(jd_keywords - resume_keywords)

    ai_suggestions = generate_ai_suggestions(resume_text, job_description, missing_skills)

    results.append({
        'filename': uploaded_resume,
        'score': round(score * 100, 2),
        'suggestions': [ai_suggestions]
    })

    return results
