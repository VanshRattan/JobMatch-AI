import os
import sys
import re
import string
import argparse
import pandas as pd
import numpy as np

# Force UTF-8 stdout if supported
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# NLP and ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer

# Ensure NLTK resources
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except Exception:
    class SimpleLemmatizer:
        def lemmatize(self, word, pos=None):
            return word
    lemmatizer = SimpleLemmatizer()

# ----------------- Skill Definitions & Taxonomies ----------------- #
SKILL_KEYWORDS = [
    # Languages
    'python', 'java', 'c++', 'c#', 'sql', 'nosql', 'javascript', 'typescript', 'rust', 'php',
    'ruby', 'swift', 'kotlin', 'scala', 'bash', 'shell', 'html', 'css',
    # Web & Fullstack
    'react', 'reactjs', 'next.js', 'nextjs', 'angular', 'vue', 'vuejs', 'node.js', 'nodejs',
    'express', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'asp.net', '.net',
    'ruby on rails', 'rails', 'laravel', 'graphql', 'rest api', 'restful', 'rest apis',
    # Data Science, AI & ML
    'machine learning', 'ml', 'deep learning', 'dl', 'nlp', 'natural language processing',
    'computer vision', 'generative ai', 'genai', 'llms', 'llm', 'artificial intelligence', 'ai',
    'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
    'opencv', 'hugging face', 'data science', 'statistics', 'predictive modeling',
    # Data Engineering & Big Data
    'spark', 'pyspark', 'hadoop', 'kafka', 'airflow', 'snowflake', 'databricks',
    'etl', 'elt', 'big data', 'data warehousing', 'data modeling', 'data pipeline',
    # Databases
    'postgresql', 'postgres', 'mysql', 'mongodb', 'redis', 'cassandra', 'dynamodb',
    'oracle', 'sqlite', 'elasticsearch', 'firebase',
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'k8s',
    'ci/cd', 'terraform', 'linux', 'git', 'github', 'gitlab', 'jenkins', 'ansible',
    # BI & Analytics
    'tableau', 'power bi', 'powerbi', 'excel', 'looker', 'data analysis', 'business intelligence',
    # Management & Soft Skills
    'agile', 'scrum', 'kanban', 'jira', 'leadership', 'communication', 'problem solving',
    'teamwork', 'critical thinking', 'project management', 'collaboration'
]

SKILL_SYNONYMS = {
    'golang': 'go',
    'k8s': 'kubernetes',
    'postgres': 'postgresql',
    'nodejs': 'node.js',
    'reactjs': 'react',
    'vuejs': 'vue',
    'nextjs': 'next.js',
    'powerbi': 'power bi',
    'sklearn': 'scikit-learn',
    'gcp': 'google cloud',
    'amazon web services': 'aws',
    'llm': 'generative ai',
    'llms': 'generative ai',
    'genai': 'generative ai',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'natural language processing': 'nlp',
    'artificial intelligence': 'ai',
    'restful': 'rest api',
    'rest apis': 'rest api',
    'pyspark': 'spark',
    'rails': 'ruby on rails',
    'github': 'git',
    'gitlab': 'git',
    'spring boot': 'spring',
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def extract_skills(raw_text):
    if not raw_text or not isinstance(raw_text, str):
        return []
    text_lower = raw_text.lower()
    found_skills = set()
    
    for skill in SKILL_KEYWORDS:
        if skill.startswith(('.', '#', '+')):
            prefix = r'(?<![a-zA-Z0-9])'
        else:
            prefix = r'\b'
            
        if skill.endswith(('+', '#')):
            suffix = r'(?![a-zA-Z0-9+#])'
        elif skill.endswith('.'):
            suffix = r'(?![a-zA-Z0-9])'
        else:
            suffix = r'\b'
            
        pattern = prefix + re.escape(skill) + suffix
        if re.search(pattern, text_lower):
            found_skills.add(SKILL_SYNONYMS.get(skill, skill))
            
    # Ambiguous short terms
    if re.search(r'\b(r\s+programming|r\s+language|r\s+studio|\bpython,\s*r\b|\br,\s*python\b)\b', text_lower):
        found_skills.add('r')
    if re.search(r'\b(golang|go\s+language|go\s+lang)\b', text_lower):
        found_skills.add('go')
        
    return sorted(list(found_skills))

def calculate_skill_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0.0
    overlap = set(resume_skills).intersection(set(jd_skills))
    return round(len(overlap) / len(set(jd_skills)), 4)

def match_single_pair(resume_text, jd_text, model=None):
    clean_res = clean_text(resume_text)
    clean_j = clean_text(jd_text)
    
    # Skills extracted from raw text to preserve symbols (C++, C#, .NET, CI/CD)
    res_skills = extract_skills(resume_text)
    j_skills = extract_skills(jd_text)
    skill_score = calculate_skill_score(res_skills, j_skills)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform([clean_res, clean_j])
    tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    
    # BERT Semantic
    bert_score = 0.0
    if model is not None:
        emb = model.encode([clean_res, clean_j], convert_to_tensor=True)
        bert_score = float(cosine_similarity(emb[0:1].cpu(), emb[1:2].cpu())[0][0])
        
    # Combined Overall Score (0 - 100%)
    if j_skills:
        combined_score = (0.40 * bert_score + 0.35 * tfidf_score + 0.25 * skill_score) * 100
    else:
        combined_score = (0.55 * bert_score + 0.45 * tfidf_score) * 100
    
    return {
        "overall_match_pct": round(combined_score, 2),
        "tfidf_score": round(tfidf_score * 100, 2),
        "bert_score": round(bert_score * 100, 2) if model else None,
        "skill_score": round(skill_score * 100, 2),
        "resume_skills": res_skills,
        "job_skills": j_skills,
        "matched_skills": list(set(res_skills).intersection(set(j_skills))),
        "missing_skills": list(set(j_skills) - set(res_skills))
    }

def run_dataset_pipeline(dataset_path="resume_job_matching_dataset.csv", sample_size=100, output_path="resume_job_matching_processed.csv"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"[INFO] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Total rows in dataset: {len(df)}")
    
    if sample_size and sample_size < len(df):
        print(f"[INFO] Using a sample of {sample_size} rows for speed (pass --all to process entire dataset)...")
        df = df.head(sample_size).copy()
    else:
        df = df.copy()
        
    print("[INFO] Preprocessing text (cleaning, stopword removal, lemmatization)...")
    df['clean_resume'] = df['resume'].apply(clean_text)
    df['clean_jd'] = df['job_description'].apply(clean_text)
    
    print("[INFO] Extracting skills...")
    df['resume_skills'] = df['clean_resume'].apply(extract_skills)
    df['jd_skills'] = df['clean_jd'].apply(extract_skills)
    df['custom_skill_score'] = df.apply(lambda x: calculate_skill_score(x['resume_skills'], x['jd_skills']), axis=1)
    
    print("[INFO] Calculating TF-IDF Cosine Similarity...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(df['clean_resume'] + " " + df['clean_jd'])
    resume_tfidf = vectorizer.transform(df['clean_resume'])
    jd_tfidf = vectorizer.transform(df['clean_jd'])
    
    df['tfidf_score'] = [cosine_similarity(resume_tfidf[i], jd_tfidf[i])[0][0] for i in range(len(df))]
    
    print("[INFO] Computing SentenceTransformer (BERT) Semantic Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embeddings = model.encode(df['clean_resume'].tolist(), convert_to_tensor=True)
    jd_embeddings = model.encode(df['clean_jd'].tolist(), convert_to_tensor=True)
    
    bert_sim = cosine_similarity(resume_embeddings.cpu(), jd_embeddings.cpu())
    df['bert_score'] = [bert_sim[i][i] for i in range(len(df))]
    
    print(f"[INFO] Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("[SUCCESS] Processed dataset saved successfully!")
    
    # Correlation & Evaluation
    corr_cols = [c for c in ['match_score', 'custom_skill_score', 'tfidf_score', 'bert_score'] if c in df.columns]
    print("\n--- Correlation Matrix ---")
    print(df[corr_cols].corr().round(4))
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume-Job Matching Runner")
    parser.add_argument("--sample", type=int, default=100, help="Number of rows to process from CSV (default: 100)")
    parser.add_argument("--all", action="store_true", help="Process all rows in dataset")
    parser.add_argument("--dataset", type=str, default="resume_job_matching_dataset.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="resume_job_matching_processed.csv", help="Output CSV path")
    args = parser.parse_args()
    
    sample_size = None if args.all else args.sample
    run_dataset_pipeline(dataset_path=args.dataset, sample_size=sample_size, output_path=args.output)
