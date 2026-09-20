import os
import re
import string
import streamlit as st
import pandas as pd
import numpy as np

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Optional PDF support
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# Streamlit Page Config
st.set_page_config(
    page_title="AI Resume-Job Matching System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .skill-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .skill-matched {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .skill-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .skill-resume {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- NLP Helper Functions ----------------- #
@st.cache_resource
def load_resources():
    try:
        stops = set(stopwords.words('english'))
    except Exception:
        nltk.download('stopwords', quiet=True)
        stops = set(stopwords.words('english'))
    
    try:
        lem = WordNetLemmatizer()
        lem.lemmatize("running")
    except Exception:
        class SimpleLem:
            def lemmatize(self, w, pos=None): return w
        lem = SimpleLem()
        
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        model = None
        
    return stops, lem, model

stop_words, lemmatizer, sbert_model = load_resources()

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

def extract_pdf_text(uploaded_file):
    if not HAS_PDF:
        return "PDF extraction library not found."
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# ----------------- UI Layout ----------------- #
st.markdown('<div class="main-header">🎯 AI-Powered Resume & Job Matching System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluate resume relevance using NLP Text Normalization, TF-IDF Cosine Similarity, Skill Extraction, and Sentence-BERT Embeddings.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration & Info")
st.sidebar.markdown("""
**Model Pipeline**:
1. **Preprocessing**: Lowercase, strip URLs & punctuation, NLTK Lemmatization & Stopword removal.
2. **Skill Match**: Keyword dictionary extraction & Jaccard overlap.
3. **TF-IDF Cosine**: Lexical keyword vector similarity.
4. **Sentence-BERT**: Semantic context embeddings (`all-MiniLM-L6-v2`).
""")

# Presets for quick testing
sample_preset = st.sidebar.selectbox(
    "Load Sample Presets:",
    [
        "None (Custom Input)",
        "Data Scientist vs ML Engineer JD (High Match)",
        "Frontend Developer vs Data Analyst JD (Low Match)",
        "Business Analyst vs Business Analyst JD (High Match)"
    ]
)

tabs = st.tabs(["🚀 Live Match Analyzer", "📊 Dataset Explorer", "📖 Documentation & How to Run"])

# Tab 1: Live Analyzer
with tabs[0]:
    col1, col2 = st.columns(2)
    
    default_resume = ""
    default_jd = ""
    
    if sample_preset == "Data Scientist vs ML Engineer JD (High Match)":
        default_resume = """Experienced Data Scientist with 4 years in Python, machine learning, deep learning, NLP, and SQL. 
Proficient in pandas, numpy, scikit-learn, TensorFlow, and PyTorch. Experience deploying models on AWS and Docker. Strong background in statistics, data science, and problem solving."""
        default_jd = """Looking for a Machine Learning Engineer with strong proficiency in Python, SQL, and Machine Learning. 
Experience with Deep Learning frameworks (TensorFlow or PyTorch), NLP, Docker, and AWS cloud deployment required. Must have good communication and teamwork skills."""
    elif sample_preset == "Frontend Developer vs Data Analyst JD (Low Match)":
        default_resume = """Frontend Developer with 3 years building responsive web interfaces using React, JavaScript, TypeScript, HTML, CSS. 
Experienced with Git, modern UI styling, and REST APIs."""
        default_jd = """Seeking a Data Analyst to join our analytics team. Required skills: SQL, Excel, Tableau, Power BI, Python, statistics, data analysis, and ETL processes."""
    elif sample_preset == "Business Analyst vs Business Analyst JD (High Match)":
        default_resume = """Business Analyst with expertise in Excel, Power BI, Tableau, SQL, data analysis, Agile, and Scrum methodologies. 
Proven track record of stakeholder communication, business reporting, and problem solving."""
        default_jd = """Hiring a Business Analyst experienced in data analysis, SQL, Excel, Tableau, Power BI. 
Knowledge of Agile and Scrum frameworks. Strong leadership and communication skills required."""
    
    with col1:
        st.subheader("📄 Candidate Resume")
        uploaded_resume = st.file_uploader("Upload Resume (.pdf or .txt) [Optional]", type=["pdf", "txt"])
        
        if uploaded_resume is not None:
            if uploaded_resume.name.endswith(".pdf"):
                extracted = extract_pdf_text(uploaded_resume)
                resume_input = st.text_area("Extracted / Editable Resume Content:", value=extracted, height=260)
            else:
                text_content = uploaded_resume.read().decode("utf-8", errors="ignore")
                resume_input = st.text_area("Extracted / Editable Resume Content:", value=text_content, height=260)
        else:
            resume_input = st.text_area(
                "Paste Resume Text Here:",
                value=default_resume,
                placeholder="Paste candidate's resume, summary, skills, experience...",
                height=260
            )

    with col2:
        st.subheader("📋 Job Description")
        jd_input = st.text_area(
            "Paste Job Description Here:",
            value=default_jd,
            placeholder="Paste role requirements, responsibilities, needed skills...",
            height=328
        )

    btn_match = st.button("⚡ Calculate Match Score", type="primary", use_container_width=True)
    
    if btn_match:
        if not resume_input.strip() or not jd_input.strip():
            st.warning("⚠️ Please provide both a Resume and a Job Description to compute matching scores.")
        else:
            with st.spinner("Analyzing text, computing TF-IDF, extracting skills, and generating BERT embeddings..."):
                clean_res = clean_text(resume_input)
                clean_j = clean_text(jd_input)
                
                # 1. Skills (Extracted from raw inputs to preserve technical symbols like C++, C#, .NET, CI/CD)
                r_skills = extract_skills(resume_input)
                j_skills = extract_skills(jd_input)
                matched_skills = sorted(list(set(r_skills).intersection(set(j_skills))))
                missing_skills = sorted(list(set(j_skills) - set(r_skills)))
                extra_skills = sorted(list(set(r_skills) - set(j_skills)))
                
                skill_score = len(matched_skills) / len(j_skills) if j_skills else 0.0
                
                # 2. TF-IDF Cosine Similarity
                vectorizer = TfidfVectorizer(max_features=5000)
                tfidf = vectorizer.fit_transform([clean_res, clean_j])
                tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
                
                # 3. Sentence-BERT Semantic Score
                if sbert_model is not None:
                    embeddings = sbert_model.encode([clean_res, clean_j], convert_to_tensor=True)
                    bert_score = float(cosine_similarity(embeddings[0:1].cpu(), embeddings[1:2].cpu())[0][0])
                else:
                    bert_score = tfidf_score
                
                # Weighted Overall Match (0 - 100%)
                # If JD specifies technical skills, blend skill overlap; otherwise weight semantic + keyword similarity fairly
                if j_skills:
                    overall_score = round((0.40 * bert_score + 0.35 * tfidf_score + 0.25 * skill_score) * 100, 1)
                else:
                    overall_score = round((0.55 * bert_score + 0.45 * tfidf_score) * 100, 1)
                
            st.markdown("---")
            st.subheader("📊 Match Results & Breakdown")
            
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    label="🎯 Overall Match Score",
                    value=f"{overall_score}%",
                    delta="High Fit" if overall_score >= 70 else ("Moderate Fit" if overall_score >= 45 else "Low Fit")
                )
            with m2:
                st.metric(label="🧠 BERT Semantic Score", value=f"{round(bert_score * 100, 1)}%")
            with m3:
                st.metric(label="🔤 TF-IDF Keyword Match", value=f"{round(tfidf_score * 100, 1)}%")
            with m4:
                st.metric(label="🛠️ Skill Overlap Score", value=f"{round(skill_score * 100, 1)}%")

            # Progress Bar
            progress_color = "normal"
            st.progress(min(max(int(overall_score), 0), 100))

            # Skills Visualization
            st.markdown("### 🧩 Skills Comparison")
            sk_col1, sk_col2, sk_col3 = st.columns(3)
            
            with sk_col1:
                st.markdown(f"**✅ Matched Skills ({len(matched_skills)}):**")
                if matched_skills:
                    badges = " ".join([f'<span class="skill-badge skill-matched">✓ {s}</span>' for s in matched_skills])
                    st.markdown(badges, unsafe_allow_html=True)
                elif not j_skills:
                    st.info("No indexed skills identified in JD.")
                else:
                    st.warning("No direct skill matches detected.")

            with sk_col2:
                st.markdown(f"**❌ Missing Skills Required by JD ({len(missing_skills)}):**")
                if not j_skills:
                    st.info("No specific technical skills detected in JD.")
                elif missing_skills:
                    badges = " ".join([f'<span class="skill-badge skill-missing">✗ {s}</span>' for s in missing_skills])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.success("All identified JD skills are present in the resume!")

            with sk_col3:
                st.markdown(f"**🌟 Additional Resume Skills ({len(extra_skills)}):**")
                if extra_skills:
                    badges = " ".join([f'<span class="skill-badge skill-resume">{s}</span>' for s in extra_skills])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("No additional skills detected.")
                    
            # Intelligent Recommendations & Actionable Insights
            st.markdown("### 💡 Intelligent Recommendations & Actionable Insights")
            
            # Executive Fit Summary
            if overall_score >= 75:
                st.success(f"🎯 **High Alignment ({overall_score}%)**: The candidate demonstrates strong relevance to the role's core expectations and technical domain.")
            elif overall_score >= 50:
                st.info(f"⚖️ **Moderate Alignment ({overall_score}%)**: The profile shows a solid foundational background, with clear opportunities for targeted refinement.")
            else:
                st.warning(f"⚠️ **Low Match Alignment ({overall_score}%)**: Noticeable divergence detected between the candidate's profile and the job requirements.")

            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("#### 🎯 Technical & Skill Optimization")
                if j_skills and missing_skills:
                    st.markdown(
                        f"""
- ❌ **Address Critical Skill Gaps**: The job description specifically requests **{len(missing_skills)}** skill(s) absent from the resume:
  - **Missing Skills**: {', '.join([f'`{s}`' for s in missing_skills])}
  - **Action**: If the candidate has hands-on, academic, or project experience with these tools, **explicitly incorporate them** into the skills section and project bullet points for ATS indexing.
  - **Bridge Strategy**: If unfamiliar with these tools, highlight transferable competencies in adjacent technologies or demonstrate ongoing coursework.
                        """
                    )
                elif j_skills and not missing_skills:
                    st.markdown(
                        f"""
- ✅ **100% Technical Skill Coverage**: All **{len(j_skills)}** technical skills identified in the job description are present in the resume!
- 🚀 **Next Level**: Focus on articulating concrete business impact using the STAR method (Situation, Task, Action, Result) with measurable metrics (e.g., latency reduction, cost savings, scale).
                        """
                    )
                else:
                    st.markdown(
                        """
- ℹ️ **General/Non-Technical JD**: The job description did not mention standard indexed technical tools.
- 🎯 **Action**: Carefully read the posting for proprietary workflows, domain terminology, or specialized methodologies, and mirror that exact phrasing directly in the resume.
                        """
                    )
                
                if extra_skills:
                    st.markdown(
                        f"""
- 🌟 **Leverage Candidate Differentiators**: Candidate brings **{len(extra_skills)}** additional skill(s) beyond the JD ({', '.join([f'`{s}`' for s in extra_skills[:6]])}):
  - **Strategy**: Position these in interviews or a cover letter as versatile value-adds for cross-functional initiatives.
                        """
                    )

            with rec_col2:
                st.markdown("#### 🔍 ATS & Semantic Alignment")
                if bert_score >= 0.65 and tfidf_score < 0.40:
                    st.markdown(
                        f"""
- ⚠️ **Semantic Relevance vs. ATS Keyword Gap**:
  - **Semantic Context Score**: `{round(bert_score * 100, 1)}%` (High)
  - **Exact Keyword Overlap**: `{round(tfidf_score * 100, 1)}%` (Low)
  - **Takeaway**: Experience is conceptually aligned, but automated ATS parsers may filter out the resume due to low exact keyword matches. **Recommendation**: Rephrase project summaries to match the exact wording used in the posting.
                        """
                    )
                elif tfidf_score >= 0.55 and bert_score < 0.40:
                    st.markdown(
                        f"""
- ⚠️ **Keyword Overlap vs. Context Depth**:
  - **Exact Keyword Match**: `{round(tfidf_score * 100, 1)}%` (High)
  - **Semantic Context Score**: `{round(bert_score * 100, 1)}%` (Low)
  - **Takeaway**: Many keywords match, but overall contextual responsibilities differ. **Recommendation**: Structure bullet points around end-to-end deliverables and business outcomes rather than isolated skill lists.
                        """
                    )
                elif overall_score >= 70:
                    st.markdown(
                        f"""
- ✨ **Harmonious Alignment**:
  - Both semantic intent (`{round(bert_score * 100, 1)}%`) and keyword density (`{round(tfidf_score * 100, 1)}%`) are well-balanced.
  - **Takeaway**: The resume is primed for both ATS screening and recruiter review.
                        """
                    )
                else:
                    st.markdown(
                        f"""
- 📉 **Broad Alignment Gap**:
  - Both semantic relevance (`{round(bert_score * 100, 1)}%`) and keyword match (`{round(tfidf_score * 100, 1)}%`) show significant divergence.
  - **Takeaway**: Consider targeting roles better matched to current strengths, or substantially restructure the resume around this specific field.
                        """
                    )
                
                st.markdown(
                    """
- 📋 **Resume Polish Tips**:
  - **Quantify Impact**: Use formulas like *"Accomplished X as measured by Y, by doing Z"*.
  - **Header Alignment**: Ensure the resume title directly mirrors the target job title.
  - **Formatting**: Keep standard ATS section headings (*Summary, Experience, Skills, Education*).
                    """
                )

# Tab 2: Dataset Explorer
with tabs[1]:
    st.subheader("📁 Dataset Exploration (`resume_job_matching_dataset.csv`)")
    data_path = "resume_job_matching_dataset.csv"
    
    if os.path.exists(data_path):
        df_preview = pd.read_csv(data_path, nrows=500)
        st.write(f"Total rows previewed: **{len(df_preview)}** (Dataset has 10,000 total rows)")
        st.dataframe(df_preview, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Provided Match Score Distribution")
            st.bar_chart(df_preview['match_score'].value_counts().sort_index())
        with c2:
            st.markdown("#### Sample Pair Inspection")
            idx = st.slider("Select Sample Index:", 0, min(100, len(df_preview) - 1), 0)
            st.markdown(f"**Target Match Score:** `{df_preview.loc[idx, 'match_score']}`")
            with st.expander("Show Job Description"):
                st.write(df_preview.loc[idx, 'job_description'])
            with st.expander("Show Resume"):
                st.write(df_preview.loc[idx, 'resume'])
    else:
        st.error(f"Dataset not found at `{data_path}`.")

# Tab 3: Documentation
with tabs[2]:
    st.subheader("📖 How to Run This Project")
    st.markdown("""
### Option 1: Run the Interactive Web Application (Recommended)
In your terminal, simply execute:
```bash
streamlit run app.py
```
This opens this web app in your browser at `http://localhost:8501`.

---

### Option 2: Run the Batch Processing Pipeline
To process the dataset and generate scores in a CSV file:
```bash
# Process a 100-sample test batch
python run_pipeline.py --sample 100

# Process all 10,000 entries
python run_pipeline.py --all
```
This will produce `resume_job_matching_processed.csv`.

---

### Option 3: Run the Jupyter Notebook
Open [`ai-powered-resume-job-matching-system.ipynb`](file:///d:/JobMatch/ai-powered-resume-job-matching-system.ipynb) in your IDE or Jupyter Lab:
1. Choose Python 3.13 kernel.
2. Click **Run All** to execute all cells, generate visualizations, and view correlation heatmaps.
    """)
