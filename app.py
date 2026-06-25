"""
app1.py — BRIEFING.AI
Streamlit frontend for an extractive legal text summarizer.
Neon-retro / cyberpunk brutalist aesthetic.
AIGC 5501 — NLP Midterm Project
Team: Isha Shah & Nirali Chaudhari
"""

import streamlit as st
from summarizer import summarize_text, preprocess_document

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BRIEFING.AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: Neon-Retro Cyberpunk Brutalist ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root Reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main {
    background-color: #050510 !important;
    color: #C8FFE8 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }

/* ── Full-width / Responsive Layout ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Orbitron', 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
h1 { font-size: 2.8rem !important; line-height: 1.1 !important; }
h3 { font-size: 1rem !important; color: #00FFB2 !important; margin-top: 1.5rem !important; }

label, p, span, div, li {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Hero Header ── */
.hero-wrap {
    border: 1px solid #00FFB222;
    border-left: 3px solid #00FFB2;
    padding: 1.4rem 1.8rem 1.2rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #050510 60%, #001a10 100%);
    position: relative;
    overflow: hidden;
    width: 100%;
    box-sizing: border-box;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00FFB2, #B400FF, transparent);
}


.hero-title {
  font-size: 2.0rem !important;
}

            
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.4rem, 5vw, 2.6rem);
    font-weight: 900;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #00FFB2 0%, #B400FF 60%, #00BFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: clamp(0.6rem, 1.5vw, 0.72rem);
    color: #555580;
    letter-spacing: 0.25em;
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    font-size: clamp(0.55rem, 1.2vw, 0.62rem);
    letter-spacing: 0.2em;
    color: #00FFB2;
    border: 1px solid #00FFB244;
    padding: 2px 10px;
    margin-top: 0.8rem;
    background: #00FFB210;
}

/* ── Section Labels ── */
.sec-label {
    font-size: 0.62rem !important;
    letter-spacing: 0.3em !important;
    color: #555580 !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #111130 !important;
    padding-bottom: 4px !important;
    margin-bottom: 0.6rem !important;
    display: block !important;
}

/* ── Textarea ── */
textarea {
    background-color: #08081A !important;
    color: #C8FFE8 !important;
    border: 1px solid #1A1A3A !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
    transition: border-color 0.2s !important;
    width: 100% !important;
}
textarea:focus {
    border-color: #00FFB266 !important;
    box-shadow: 0 0 12px #00FFB222 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1A1A3A !important;
    border-radius: 0 !important;
    padding: 8px !important;
    background: #08081A !important;
    width: 100% !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border-radius: 0 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: #00FFB2 !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    color: #C8FFE8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
}

/* ── Primary Button ── */
[data-testid="stButton"] > button {
    background: transparent !important;
    color: #00FFB2 !important;
    border: 1px solid #00FFB2 !important;
    border-radius: 0 !important;
    padding: 0.65rem 1.4rem !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 0 16px #00FFB220 !important;
}
[data-testid="stButton"] > button:hover {
    background: #00FFB2 !important;
    color: #050510 !important;
    box-shadow: 0 0 30px #00FFB260 !important;
}

/* ── Metrics Row ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 1rem 0 1.4rem;
    width: 100%;
    box-sizing: border-box;
}
.metric-card {
    background: #08081A;
    border: 1px solid #111130;
    border-top: 2px solid #00FFB2;
    padding: 14px 18px;
    box-sizing: border-box;
}
.metric-label {
    font-size: 0.58rem;
    letter-spacing: 0.3em;
    color: #555580;
    text-transform: uppercase;
    display: block;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1rem, 3vw, 1.5rem);
    font-weight: 700;
    color: #00FFB2;
    line-height: 1;
    word-break: break-all;
}
.metric-value.accent { color: #B400FF; }
.metric-arrow { color: #555580; font-size: 1rem; }

/* ── Summary Output ── */
.summary-shell {
    border: 1px solid #111130;
    border-left: 3px solid #B400FF;
    background: #06060F;
    padding: 0;
    margin-top: 0.6rem;
    position: relative;
    overflow: hidden;
    width: 100%;
    box-sizing: border-box;
}
.summary-shell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #B400FF, transparent);
}
.summary-item {
    padding: 12px 18px;
    border-bottom: 1px solid #0D0D22;
    font-size: clamp(0.78rem, 2vw, 0.88rem);
    color: #C8FFE8;
    line-height: 1.6;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    box-sizing: border-box;
}
.summary-item:last-child { border-bottom: none; }
.summary-num {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    color: #B400FF;
    min-width: 22px;
    padding-top: 3px;
    letter-spacing: 0.1em;
    flex-shrink: 0;
}

/* ── Alerts ── */
div.stAlert {
    background: #08081A !important;
    border: 1px solid #1A1A3A !important;
    border-radius: 0 !important;
    color: #C8FFE8 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #0D0D22 !important;
    margin: 1.4rem 0 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #222244;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #0D0D22;
    text-transform: uppercase;
}
.footer span { color: #00FFB240; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #00FFB2 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050510; }
::-webkit-scrollbar-thumb { background: #00FFB240; border-radius: 0; }

/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-wrap {
        padding: 1rem 1.2rem;
    }
    .metrics-grid {
        grid-template-columns: 1fr !important;
    }
}
            
/* ── File Uploader Button Fix ── */
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0 !important;
    color: transparent !important;
    position: relative !important;
    min-width: 120px !important;
    min-height: 36px !important;
    padding: 0.5rem 1.2rem !important;
    background: transparent !important;
    border: 1px solid #00FFB2 !important;
    border-radius: 0 !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] button span {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: 'Upload' !important;
    font-size: 0.75rem !important;
    color: #00FFB2 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.15em !important;
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
}

/* Upload button fix */    
            
[data-testid="stFileUploader"] button[kind="borderlessIcon"],
[data-testid="stFileUploader"] .stFileUploader ~ div button,
[data-testid="stFileUploaderDeleteBtn"] {
    display: none !important;
}
            
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #00FFB2 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: #050510 !important;
}
</style>
            
            
""", unsafe_allow_html=True)

st.markdown("""
<script>
const hideUploadBtn = () => {
    document.querySelectorAll('[data-testid="stFileUploader"] button').forEach((btn, i) => {
        if (i > 0) btn.style.display = 'none';
    });
};
const observer = new MutationObserver(hideUploadBtn);
observer.observe(document.body, { childList: true, subtree: true });
hideUploadBtn();
</script>
""", unsafe_allow_html=True)


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">BRIEFING.AI</p>
    <p class="hero-sub">Extractive Legal Text Summarization Engine &nbsp;·&nbsp; Classical NLP + PageRank</p>
    <!-- <span class="hero-badge">⚡ AIGC 5501 &nbsp;·&nbsp; NLP // TEXT SUMMARIZATION &nbsp;·&nbsp; ISHA SHAH &amp; NIRALI CHAUDHARI</span> -->
</div>
""", unsafe_allow_html=True)

# ── Input Section ──────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">01 &nbsp;/&nbsp; Input Payload</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload document",
    type=["txt", "pdf", "docx"],
    help="Upload a TXT, PDF, or DOCX legal / Terms of Service document."
)

document_text = ""
if uploaded_file is not None:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".txt"):
            document_text = uploaded_file.read().decode("utf-8")

        elif fname.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                document_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                st.warning("PDF support requires pypdf. Run: pip install pypdf")

        elif fname.endswith(".docx"):
            try:
                import docx
                import io
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                document_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                st.warning("DOCX support requires python-docx. Run: pip install python-docx")

        if document_text.strip():
            st.success(f"FILE LOADED — {uploaded_file.name} — {len(document_text):,} characters")
        else:
            st.error("ERROR — Could not extract text from this file.")
    except Exception as e:
        st.error(f"READ ERROR: {str(e).upper()}")

input_text = st.text_area(
    "Paste legal payload",
    value=document_text if document_text else "",
    height=220,
    placeholder="Paste terms of service, privacy policy, or contract text here...\n\nThe engine will tokenize, lemmatize, and run TF-IDF + PageRank centrality\nto extract the most important sentences.",
    label_visibility="collapsed"
)

if input_text.strip():
    char_count = len(input_text)
    word_count = len(input_text.split())
    st.caption(f"↳  {char_count:,} chars &nbsp;·&nbsp; {word_count:,} words")

st.markdown("---")

# ── Configuration ──────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">02 &nbsp;/&nbsp; Summary Configuration</span>', unsafe_allow_html=True)

cfg_col1, cfg_col2 = st.columns([1, 1])

with cfg_col1:
    summary_mode = st.radio(
        "Summary length mode",
        options=["SENTENCE COUNT", "COMPRESSION RATIO"],
        horizontal=True,
        label_visibility="collapsed"
    )

with cfg_col2:
    if summary_mode == "SENTENCE COUNT":
        target_sentences = st.slider("Target sentences", min_value=1, max_value=25, value=5, step=1)
        st.caption(f"Extract {target_sentences} most central sentence{'s' if target_sentences > 1 else ''}")
    else:
        compression_ratio = st.slider("Compression ratio (%)", min_value=5, max_value=50, value=20, step=5)
        st.caption(f"Retain {compression_ratio}% of the original sentence count")

st.markdown("---")

# ── Run Button ─────────────────────────────────────────────────────────────────
run = st.button("▶  RUN SUMMARIZATION ENGINE")

# ── Engine Logic ───────────────────────────────────────────────────────────────
if run:
    if not input_text.strip():
        st.error("ERROR — TEXT PAYLOAD IS EMPTY. PROVIDE INPUT TEXT ABOVE.")
    else:
        try:
            sentences_data = preprocess_document(input_text)
            total_sentences = len(sentences_data)

            if total_sentences == 0:
                st.error("ERROR — NO PARSABLE SENTENCES DETECTED IN INPUT.")
            else:
                if summary_mode == "SENTENCE COUNT":
                    target_count = target_sentences
                else:
                    target_count = max(1, int(total_sentences * (compression_ratio / 100.0)))

                with st.spinner("ANALYZING SENTENCE GRAPH · COMPUTING PAGERANK CENTRALITY..."):
                    summary = summarize_text(input_text, target_count=target_count)

                # ── Metrics ────────────────────────────────────────────────────
                st.markdown('<span class="sec-label">03 &nbsp;/&nbsp; Engine Metrics</span>', unsafe_allow_html=True)

                orig_words = len(input_text.split())
                summ_words = len(" ".join(summary).split())
                compression_pct = (1.0 - (summ_words / max(1, orig_words))) * 100

                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <span class="metric-label">Sentences</span>
                        <span class="metric-value">{total_sentences} <span class="metric-arrow">→</span> {len(summary)}</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Word Count</span>
                        <span class="metric-value">{orig_words:,} <span class="metric-arrow">→</span> {summ_words:,}</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Reduction</span>
                        <span class="metric-value accent">{compression_pct:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Summary Output ─────────────────────────────────────────────
                st.markdown('<span class="sec-label">04 &nbsp;/&nbsp; Extracted Summary — Chronological Sequence</span>', unsafe_allow_html=True)

                items_html = ""
                for i, sentence in enumerate(summary, 1):
                    num = f"[{i:02d}]"
                    items_html += f'<div class="summary-item"><span class="summary-num">{num}</span><span>{sentence}</span></div>'

                st.markdown(f'<div class="summary-shell">{items_html}</div>', unsafe_allow_html=True)

                # ── Algorithm Note ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<span class="sec-label">05 &nbsp;/&nbsp; Pipeline Trace</span>', unsafe_allow_html=True)
                st.caption(
                    f"spaCy tokenization + lemmatization  →  TF-IDF sparse matrix ({total_sentences} sentences)  "
                    f"→  Cosine similarity graph  →  PageRank (α=0.85)  →  Top {len(summary)} nodes extracted chronologically"
                )

        except Exception as e:
            st.error(f"EXECUTION FAILURE — {str(e).upper()}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    BRIEFING.AI &nbsp;·&nbsp; <span>Pure NLP · No LLM Dependencies · Secure Local Analysis</span>
    &nbsp;·&nbsp; AIGC 5501 &nbsp;·&nbsp; Isha Shah &amp; Nirali Chaudhari
</div>
""", unsafe_allow_html=True)