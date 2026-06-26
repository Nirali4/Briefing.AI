# BRIEFING.AI

Extractive text summarization engine tailored for long, complex terms-of-service and legal documents. Built using classical Natural Language Processing (NLP), Machine Learning graph centrality, and a hyper-minimalist, brutalist Streamlit interface.

📊 **Project Presentation**: [Download PDF Presentation](assets/Briefing.AI_Presentation.pdf) (If the GitHub preview fails, download/view the raw PDF file directly).

---

## FEATURES
- **Pure Classical NLP & ML**: Sentence tokenization and lemmatization via `spaCy`, TF-IDF vectorization, Cosine Similarity matrices, and `PageRank` scoring (via `NetworkX`).
- **No LLM/Transformer API Calls**: Zero external model dependencies or api-keys; runs 100% locally and securely.
- **Custom Legal Preprocessing**: Specialized stop word filters for legalese (e.g., *whereas*, *hereby*, *heretofore*, *covenants*).
- **Brutalist Monochrome Aesthetic**: Stark pitch black interface with silver/white monospace text, professional metrics, and zero visual clutter.
- **Flexible Controls**: Select summary lengths by absolute sentence count or target compression ratio.
- **Detailed Evaluation Framework**: Built-in benchmark suite analyzing execution time, sentence compression ratios, PageRank score distributions, and sentence connection topology graph structures.

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
# Additional libraries for evaluation, plotting, and reading files:
pip install matplotlib pandas seaborn pypdf python-docx ipykernel
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

Or, if .venv is not activated
```bash
.venv\Scripts\streamlit.exe run app.py
```

This will start the local server and open the web app in your browser (usually at `http://localhost:8501`).

### 2. Run Summarization
- **Input Text**: Either upload a plain text, PDF, or DOCX legal document using the uploader, or copy-paste the contract/TOS text directly into the primary input area.
- **Configure Length**: Use the configuration controls to choose between **Sentence Count** or **Compression Ratio** and adjust the slider to your desired summary length.
- **Execute**: Click **RUN SUMMARIZATION ENGINE** to compute sentence centrality. The top N central sentences will be extracted and presented in their original chronological sequence alongside compression metrics.

---

## PERFORMANCE & EVALUATION

A comprehensive quantitative evaluation suite has been created to analyze the summarizer's accuracy, execution efficiency, and centrality distribution across different legal payloads:

- **Jupyter Notebook**: [Evaluation_and_Results.ipynb](Evaluation_and_Results.ipynb) details dataset profile metrics, legal preprocessing challenges, execution benchmarks, and network graphs.
- **Benchmark Data**: [performance_metrics.csv](assets/performance_metrics.csv) contains character, word, sentence profiles and processing times for the files in the `test/` folder.
- **Generated Visualizations** (located in the `assets/` directory):
  - `performance_plot.png`: Line plot representing document scale (sentence count) vs. preprocessing and algorithmic execution times.
  - `compression_plot.png`: Bar chart comparing original word counts to summarized word counts.
  - `pagerank_distribution.png`: Score distribution indicating selection boundary clarity.
  - `sentence_network.png`: Topological graph displaying connectivity between sentences, highlighting top central nodes in green.

To re-run the benchmark suite and generate the notebook outputs:
```bash
python scratch/run_evaluation.py
```

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
