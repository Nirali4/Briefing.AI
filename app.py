"""
app.py

Streamlit frontend for Briefing.ai, an extractive legal text summarizer.
Implements a hyper-minimalist, brutalist, monochrome user interface.
"""

import streamlit as st
from summarizer import summarize_text, preprocess_document

# Set Page Config
st.set_page_config(
    page_title="BRIEFING.AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Brutalist Monochrome CSS Injection
st.markdown(
    """
    <style>
    /* Strict Pitch Black Background & White Monospace Text */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #000000 !important;
        color: #E0E0E0 !important;
        font-family: "Courier New", Courier, monospace !important;
    }
    
    /* Header removal/transparency */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        border-bottom: none !important;
    }

    /* Style titles and headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: "Courier New", Courier, monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Professional dividing lines */
    hr {
        border: 0 !important;
        border-top: 1px solid #222222 !important;
        margin: 1.5rem 0 !important;
    }

    /* Remove padding/margins for brutalist look */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 900px !important;
    }

    /* Style Text Area */
    textarea {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
        font-family: "Courier New", Courier, monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
        padding: 10px !important;
    }
    textarea:focus {
        border-color: #FFFFFF !important;
        box-shadow: none !important;
    }

    /* Style Inputs & Labels */
    label, p, span, div {
        font-family: "Courier New", Courier, monospace !important;
        color: #CCCCCC !important;
        font-size: 0.85rem !important;
    }

    /* File Uploader custom border */
    [data-testid="stFileUploader"] {
        border: 1px dashed #333333 !important;
        border-radius: 0px !important;
        padding: 10px !important;
        background-color: #000000 !important;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 0px !important;
        background-color: #000000 !important;
    }

    /* Dark Slate Panels for info boxes */
    div.stAlert, div[data-testid="stMetricContainer"] {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 0px !important;
        color: #FFFFFF !important;
        padding: 15px !important;
    }

    /* Streamlit buttons */
    button, [role="button"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
        border-radius: 0px !important;
        padding: 0.5rem 1rem !important;
        font-family: "Courier New", Courier, monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
    }

    /* Footer styling */
    .brutalist-footer {
        font-size: 0.75rem !important;
        color: #444444 !important;
        text-align: center;
        margin-top: 4rem;
        letter-spacing: 0.1em;
    }
    
    /* Code/output block */
    .summary-box {
        background-color: #050505 !important;
        border-left: 2px solid #FFFFFF !important;
        padding: 15px !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
    }
    .summary-sentence {
        margin-bottom: 12px;
        line-height: 1.5;
        font-size: 0.95rem;
        color: #E0E0E0;
    }
    .stat-val {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application Header
st.title("BRIEFING.AI")
st.text("EXTRACTIVE LEGAL TEXT SUMMARIZATION ENGINE // CLASSICAL NLP")
st.markdown("---")

# Main document payload placeholder
document_text = ""

# File uploader (Secondary component)
uploaded_file = st.file_uploader(
    "UPLOAD DOCUMENT (.TXT)", 
    type=["txt"], 
    help="Upload a plain text legal or terms-of-service document."
)

if uploaded_file is not None:
    try:
        document_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"ERROR READING UPLOADED FILE: {str(e).upper()}")

# Large Text Input Area (Primary component)
input_text = st.text_area(
    "PASTE LEGAL PAYLOAD HERE",
    value=document_text if document_text else "",
    height=250,
    placeholder="PASTE TERMS OF SERVICE, PRIVACY POLICY, OR LEGAL DOCUMENT CONTRACT TEXT..."
)

st.markdown("---")

# Slider controls
col1, col2 = st.columns([2, 1])

with col1:
    summary_mode = st.radio(
        "SUMMARY LENGTH MODE",
        options=["SENTENCE COUNT", "COMPRESSION RATIO"],
        horizontal=True
    )

with col2:
    if summary_mode == "SENTENCE COUNT":
        target_sentences = st.slider("TARGET SENTENCES", min_value=1, max_value=25, value=5, step=1)
    else:
        compression_ratio = st.slider("COMPRESSION RATIO (%)", min_value=5, max_value=50, value=20, step=5)

# Process trigger button
if st.button("RUN SUMMARIZATION ENGINE"):
    if not input_text.strip():
        st.error("ERROR: TEXT PAYLOAD IS EMPTY. PLEASE PROVIDE INPUT TEXT.")
    else:
        try:
            # First, check total sentences to calculate ratio if needed
            sentences_data = preprocess_document(input_text)
            total_sentences = len(sentences_data)
            
            if total_sentences == 0:
                st.error("ERROR: NO PARSABLE SENTENCES DETECTED.")
            else:
                # Calculate target count based on mode
                if summary_mode == "SENTENCE COUNT":
                    target_count = target_sentences
                else:
                    target_count = max(1, int(total_sentences * (compression_ratio / 100.0)))
                
                # Run summarizer backend
                with st.spinner("ANALYZING MATRIX CENTRALITY..."):
                    summary = summarize_text(input_text, target_count=target_count)
                
                # Metrics Display
                st.markdown("### ENGINE METRICS")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                # Original word count vs summarized word count
                orig_words = len(input_text.split())
                summ_words = len(" ".join(summary).split())
                compression_pct = (1.0 - (summ_words / max(1, orig_words))) * 100
                
                with stat_col1:
                    st.markdown(f"SENTENCES<br><span class='stat-val'>{total_sentences} &rarr; {len(summary)}</span>", unsafe_allow_html=True)
                with stat_col2:
                    st.markdown(f"WORD COUNT<br><span class='stat-val'>{orig_words} &rarr; {summ_words}</span>", unsafe_allow_html=True)
                with stat_col3:
                    st.markdown(f"REDUCTION RATIO<br><span class='stat-val'>{compression_pct:.1f}%</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Summary Output
                st.markdown("### EXTRACTED SUMMARY (CHRONOLOGICAL SEQUENCE)")
                
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                for sentence in summary:
                    st.markdown(f'<div class="summary-sentence">&bull; {sentence}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"EXECUTION FAILURE: {str(e).upper()}")

# Footer
st.markdown(
    """
    <div class="brutalist-footer">
        BRIEFING.AI // PURE NLP ENGINE // NO LLM DEPENDENCIES // SECURE LOCAL ANALYSIS
    </div>
    """,
    unsafe_allow_html=True
)
