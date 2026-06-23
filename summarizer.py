"""
summarizer.py

Algorithmic engine for extractive text summarization of long, complex legal documents.
Utilizes pure classical NLP:
1. Sentence segmentation and lemmatization via spaCy.
2. Custom stop word filtering (standard + legal-specific terms).
3. TF-IDF vectorization.
4. Cosine similarity matrix generation.
5. PageRank graph centrality scoring (via NetworkX).
6. Chronological extraction of top-ranked sentences.
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load spaCy pipeline (disable parser and ner for faster processing where possible,
# but keep sentencizer/dependency parser for high-quality sentence boundary detection)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to load default blank English model.")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

# Define legal-specific stop words to filter out during vectorization/similarity computation
LEGAL_STOP_WORDS = {
    "whereas", "heretofore", "hereby", "herein", "therein", "hereof", "thereof",
    "hereinafter", "aforesaid", "said", "such", "witnesseth", "agreement", "party",
    "parties", "contract", "covenant", "hereto", "hereunder", "thereunder", "therefrom",
    "hereinabove", "thereto", "hereinbefore", "thereinbefore", "whereby"
}

def preprocess_document(text: str) -> List[Tuple[int, str, str]]:
    """
    Segment the document into sentences and preprocess each sentence.
    
    Args:
        text (str): The raw input text.
        
    Returns:
        List[Tuple[int, str, str]]: A list of tuples containing:
            - int: The original chronological index of the sentence.
            - str: The original sentence text.
            - str: The preprocessed/lemmatized sentence text.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences_data = []
    
    # Process sentence by sentence
    for idx, sent in enumerate(doc.sents):
        original_text = sent.text.strip()
        if not original_text:
            continue
            
        # Clean, lemmatize, lowercase, and filter stop words
        cleaned_tokens = []
        for token in sent:
            # Check if token is alphabetic, not a standard stop word, and not a legal stop word
            token_text_lower = token.text.lower()
            if (token.is_alpha and 
                not token.is_stop and 
                token_text_lower not in LEGAL_STOP_WORDS and 
                len(token.lemma_) > 1):
                cleaned_tokens.append(token.lemma_.lower())
        
        preprocessed_text = " ".join(cleaned_tokens)
        sentences_data.append((idx, original_text, preprocessed_text))
        
    return sentences_data

def summarize_text(text: str, target_count: int = 5) -> List[str]:
    """
    Extract the top N central sentences from the text, ordered chronologically.
    
    Args:
        text (str): The raw legal document text.
        target_count (int): Number of sentences to extract for the summary.
        
    Returns:
        List[str]: The summary sentences in chronological order.
    """
    # 1. Preprocess and segment sentences
    sentences_data = preprocess_document(text)
    if not sentences_data:
        raise ValueError("The provided text contains no valid sentences after parsing.")
    
    total_sentences = len(sentences_data)
    logger.info(f"Total parsed sentences: {total_sentences}")
    
    # If the document has fewer sentences than requested, return all sentences in order
    if total_sentences <= target_count:
        logger.info(f"Requested count ({target_count}) is greater than or equal to total sentences ({total_sentences}). Returning full text.")
        return [sent[1] for sent in sentences_data]
    
    # Extract preprocessed texts for TF-IDF computation
    preprocessed_sents = [data[2] for data in sentences_data]
    
    # Check if there's any content left after preprocessing (e.g. not all blank or stop words)
    valid_content_count = sum(1 for s in preprocessed_sents if s.strip())
    if valid_content_count == 0:
        # Fall back to using the original lowercase text if everything was filtered out
        logger.warning("All tokens filtered as stop words. Falling back to simple lowercased tokens.")
        preprocessed_sents = [data[1].lower() for data in sentences_data]

    # 2. Build Sentence Similarity Matrix via TF-IDF + Cosine Similarity
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sents)
    except ValueError as e:
        logger.error(f"TF-IDF vectorizer initialization failed: {e}")
        # Fall back to returning the first N sentences if vectorization fails
        return [sent[1] for sent in sentences_data[:target_count]]

    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 3. Apply PageRank over the similarity matrix
    # Build NetworkX graph from similarity matrix
    nx_graph = nx.Graph()
    for i in range(total_sentences):
        for j in range(i + 1, total_sentences):
            weight = similarity_matrix[i][j]
            # Avoid inserting negligible weights or self-loops to maintain centrality accuracy
            if weight > 1e-4:
                nx_graph.add_edge(i, j, weight=weight)
                
    # If the graph is completely disconnected/empty, PageRank cannot be computed.
    # Fall back to using TF-IDF sum or returning top N chronological sentences.
    if nx_graph.number_of_nodes() == 0:
        logger.warning("Graph has zero edges. Falling back to sequential baseline summary.")
        return [sent[1] for sent in sentences_data[:target_count]]
        
    try:
        # Run PageRank centrality algorithm
        # pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
        scores = nx.pagerank(nx_graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank power iteration failed to converge. Falling back to degree centrality.")
        scores = {node: sum(data.get("weight", 0) for _, data in nx_graph[node].items()) for node in nx_graph.nodes()}
    except Exception as e:
        logger.error(f"Unexpected error during PageRank: {e}")
        scores = {i: 0.0 for i in range(total_sentences)}
        
    # Map scores back to all sentences (nodes not in graph get score 0)
    full_scores: Dict[int, float] = {i: scores.get(i, 0.0) for i in range(total_sentences)}
    
    # 4. Extract top N sentences and sort them chronologically
    # Sort sentences by PageRank score in descending order
    ranked_indices = sorted(full_scores.keys(), key=lambda idx: full_scores[idx], reverse=True)
    top_indices = ranked_indices[:target_count]
    
    # Order the selected top indices chronologically
    top_indices_chronological = sorted(top_indices)
    
    summary_sentences = [sentences_data[i][1] for i in top_indices_chronological]
    return summary_sentences
