# BRIEFING.AI

Extractive text summarization engine tailored for long, complex terms-of-service and legal documents. Built using classical Natural Language Processing (NLP), Machine Learning graph centrality, and a hyper-minimalist, brutalist Streamlit interface.

---

## FEATURES
- **Pure Classical NLP & ML**: Sentence tokenization and lemmatization via `spaCy`, TF-IDF vectorization, Cosine Similarity matrices, and `PageRank` scoring (via `NetworkX`).
- **No LLM/Transformer API Calls**: Zero external model dependencies or api-keys; runs 100% locally and securely.
- **Custom Legal Preprocessing**: Specialized stop word filters for legalese (e.g., *whereas*, *hereby*, *heretofore*, *covenants*).
- **Brutalist Monochrome Aesthetic**: Stark pitch black interface with silver/white monospace text, professional metrics, and zero visual clutter.
- **Flexible Controls**: Select summary lengths by absolute sentence count or target compression ratio.

---

## QUICK SETUP

### 1. Clone the Repository
```bash
git clone https://github.com/Nirali4/Briefing.AI.git
cd Briefing.AI
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Language Pipeline
```bash
python -m spacy download en_core_web_sm
```

---

## HOW TO USE

### 1. Launch the Streamlit Server
```bash
streamlit run app.py
```
This will start the local server and open the web app in your browser (usually at `http://localhost:8501`).

### 2. Run Summarization
- **Input Text**: Either upload a plain text legal document (`.txt`) using the file uploader or copy-paste the contract/TOS text directly into the primary input area.
- **Configure Length**: Use the configuration controls to choose between **Sentence Count** or **Compression Ratio** and adjust the slider to your desired summary length.
- **Execute**: Click **RUN SUMMARIZATION ENGINE** to compute sentence centrality. The top N central sentences will be extracted and presented in their original chronological sequence alongside compression metrics.

---

## SYSTEM ARCHITECTURE
```
[Input Document] 
      │
      ▼
[spaCy Pipeline] ──► Sentence Tokenization & Custom Stop Words Filtering
      │
      ▼
[scikit-learn]   ──► TF-IDF Sparse Matrix Construction
      │
      ▼
[Cosine Sim]     ──► Matrix Pairwise Cosine Similarity Computation
      │
      ▼
[NetworkX]       ──► PageRank centrality computed over Sentence Similarity Graph
      │
      ▼
[Sort & Output]  ──► Extraction of Top N nodes mapped back chronologically
```
