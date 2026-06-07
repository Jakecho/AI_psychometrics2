"""
NCLEX-RN Vector Database Search System (v3)
Copied from NCLEX_RN_VectorDB_v2.py
"""

import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from typing import List, Dict, Any
import numpy as np
import os

# ==================== Configuration ====================
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "007000",
    "host": "localhost",
    "port": 5432
}

# Use CSV instead of connecting to Postgres for v3
USE_CSV = True
CSV_PATH = "item_bank_hosted.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SIMILARITY_EPSILON = 1e-12
RESULT_COLUMNS = [
    'item_id', 'domain', 'topic', 'stem',
    'choice_A', 'choice_B', 'choice_C', 'choice_D',
    'key', 'rationale', 'rasch_b', 'pvalue',
    'point_biserial', 'enemy', 'similarity_score'
]

# ==================== Helper Functions ====================

@st.cache_data
def load_csv_data(path: str = CSV_PATH):
    """Load item-bank CSV and parse the stored embedding column."""
    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), None)

    # Merge enemy column from item_bank_clean.csv if missing in hosted CSV
    if 'enemy' not in df.columns and os.path.exists('item_bank_clean.csv'):
        try:
            clean_df = pd.read_csv('item_bank_clean.csv', usecols=['item_id', 'enemy'])
            df['item_id_str'] = df['item_id'].astype(str)
            clean_df['item_id_str'] = clean_df['item_id'].astype(str)
            df = df.merge(clean_df[['item_id_str', 'enemy']], on='item_id_str', how='left')
            df.drop(columns=['item_id_str'], inplace=True)
            df = df.where(pd.notnull(df), None)
        except Exception:
            pass

    emb_list = []
    for v in df.get('embedding', []):
        if isinstance(v, str):
            try:
                arr = np.array(json.loads(v), dtype=float)
            except Exception:
                arr = np.array([])
        elif v is None:
            arr = np.array([])
        else:
            arr = np.array(v, dtype=float)
        emb_list.append(arr)

    emb_matrix = np.zeros((len(df), EMBEDDING_DIM), dtype=float)
    for i, arr in enumerate(emb_list):
        if arr.size == 0:
            continue
        if arr.size != EMBEDDING_DIM:
            arr = arr[:EMBEDDING_DIM]
            if arr.size < EMBEDDING_DIM:
                padded = np.zeros(EMBEDDING_DIM)
                padded[:arr.size] = arr
                arr = padded
        emb_matrix[i] = arr

    return df, emb_matrix


def build_item_text(row) -> str:
    """Build searchable text from CSV columns when a DB-only combined field is absent."""
    text_fields = [
        'stem', 'choice_A', 'choice_B', 'choice_C', 'choice_D',
        'rationale', 'domain', 'topic'
    ]
    parts = []
    for field in text_fields:
        value = row.get(field)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def row_to_item_tuple(row):
    """Return an item tuple compatible with the existing preview code."""
    combined = row.get('combined')
    if combined is None or not str(combined).strip():
        combined = build_item_text(row)

    return (
        row.get('item_id'), row.get('domain'), row.get('topic'), row.get('stem'),
        row.get('choice_A'), row.get('choice_B'), row.get('choice_C'), row.get('choice_D'),
        row.get('key'), row.get('rationale'), row.get('rasch_b'), row.get('pvalue'),
        row.get('point_biserial'), combined
    )


def csv_status():
    """Return CSV availability details for the sidebar."""
    if not os.path.exists(CSV_PATH):
        return None, f"CSV data source not found: {CSV_PATH}"
    try:
        df, emb_matrix = load_csv_data(CSV_PATH)
        embedding_count = int(np.count_nonzero(np.linalg.norm(emb_matrix, axis=1)))
        return {
            "rows": len(df),
            "embedding_count": embedding_count,
            "path": CSV_PATH
        }, None
    except Exception as e:
        return None, str(e)


def apply_result_filter(results_df: pd.DataFrame, filtering_mode: str, top_k: int, top_p: float) -> pd.DataFrame:
    """Apply the selected Top-K / Top-P filter to a search result frame."""
    if results_df.empty:
        return results_df

    filtered = results_df.copy()
    min_similarity = top_p - SIMILARITY_EPSILON
    if filtering_mode == "Top P only":
        filtered = filtered[filtered['similarity_score'] >= min_similarity]
    elif filtering_mode == "Both Top K and Top P":
        filtered = filtered.head(top_k)
        filtered = filtered[filtered['similarity_score'] >= min_similarity]
    elif filtering_mode == "Top K only":
        filtered = filtered.head(top_k)

    return filtered.reset_index(drop=True)


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    return SentenceTransformer(EMBEDDING_MODEL)

def get_db_connection():
    """Create database connection (disabled when USE_CSV=True)"""
    if USE_CSV:
        return None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

def generate_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """Generate embedding for input text"""
    embedding = model.encode([text], normalize_embeddings=True)[0]
    return embedding.tolist()

def vector_search(query_embedding: List[float], top_k: int = 5, search_method: str = "semantic", 
                  keywords: str = None, semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> pd.DataFrame:
    """
    Perform vector similarity search. Supports CSV fallback when USE_CSV is True.
    """
    # If using CSV data source, compute similarities locally
    if USE_CSV:
        if not os.path.exists(CSV_PATH):
            st.error(f"❌ CSV data source not found: {CSV_PATH}")
            return pd.DataFrame()

        df, emb_matrix = load_csv_data(CSV_PATH)

        # Normalize embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        emb_normed = emb_matrix / norms

        q = np.array(query_embedding)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            qn = q
        else:
            qn = q / q_norm

        sims = np.clip(emb_normed.dot(qn), -1.0, 1.0)

        # Prepare keyword score if needed (simple token count)
        if search_method in ["hybrid_weighted", "rrf"]:
            kw = keywords if keywords and keywords.strip() else ''
            kw_tokens = [t for t in kw.lower().split() if t]
            text_corpus = df.apply(build_item_text, axis=1).str.lower()
            if len(kw_tokens) == 0:
                keyword_scores = np.zeros(len(df))
            else:
                keyword_scores = np.array([sum(text.count(tok) for tok in kw_tokens) for text in text_corpus])
            # normalize keyword scores
            if keyword_scores.max() > 0:
                keyword_scores = keyword_scores / keyword_scores.max()
            else:
                keyword_scores = np.zeros_like(keyword_scores)

        # Compute results depending on strategy
        if search_method == 'semantic':
            idx = np.argsort(sims)[::-1][:top_k]
            rows = []
            for i in idx:
                row = df.iloc[i]
                score = float(sims[i])
                rows.append((row.get('item_id'), row.get('domain'), row.get('topic'), row.get('stem'),
                             row.get('choice_A'), row.get('choice_B'), row.get('choice_C'), row.get('choice_D'),
                             row.get('key'), row.get('rationale'), row.get('rasch_b'), row.get('pvalue'),
                             row.get('point_biserial'), row.get('enemy'), score))
            return pd.DataFrame(rows, columns=RESULT_COLUMNS)

        elif search_method == 'hybrid_weighted':
            # combine keyword_scores and sims with weights
            total = semantic_weight + keyword_weight
            if total == 0:
                sw = 0.5
                kw = 0.5
            else:
                sw = semantic_weight / total
                kw = keyword_weight / total
            combined = sw * sims + kw * keyword_scores
            idx = np.argsort(combined)[::-1][:top_k]
            rows = []
            for i in idx:
                row = df.iloc[i]
                rows.append((row.get('item_id'), row.get('domain'), row.get('topic'), row.get('stem'),
                             row.get('choice_A'), row.get('choice_B'), row.get('choice_C'), row.get('choice_D'),
                             row.get('key'), row.get('rationale'), row.get('rasch_b'), row.get('pvalue'),
                             row.get('point_biserial'), row.get('enemy'), float(combined[i])))
            return pd.DataFrame(rows, columns=RESULT_COLUMNS)

        elif search_method == 'rrf':
            # Reciprocal Rank Fusion using ranks from keyword and semantic
            k = 60
            # semantic ranks (1 is best)
            sem_ranks = np.argsort(sims)[::-1]
            sem_rank_pos = np.empty(len(sims), dtype=int)
            sem_rank_pos[sem_ranks] = np.arange(1, len(sims)+1)
            # keyword ranks
            if search_method in ["hybrid_weighted", "rrf"]:
                kw_ranks = np.argsort(keyword_scores)[::-1]
                kw_rank_pos = np.empty(len(keyword_scores), dtype=int)
                kw_rank_pos[kw_ranks] = np.arange(1, len(keyword_scores)+1)
            else:
                kw_rank_pos = np.full(len(sims), len(sims)+1)

            rrf_scores = (1.0 / (k + kw_rank_pos)) + (1.0 / (k + sem_rank_pos))
            idx = np.argsort(rrf_scores)[::-1][:top_k]
            rows = []
            for i in idx:
                row = df.iloc[i]
                rows.append((row.get('item_id'), row.get('domain'), row.get('topic'), row.get('stem'),
                             row.get('choice_A'), row.get('choice_B'), row.get('choice_C'), row.get('choice_D'),
                             row.get('key'), row.get('rationale'), row.get('rasch_b'), row.get('pvalue'),
                             row.get('point_biserial'), row.get('enemy'), float(rrf_scores[i])))
            return pd.DataFrame(rows, columns=RESULT_COLUMNS)

        else:
            # default to semantic
            return vector_search(query_embedding, top_k, 'semantic')

    # If not using CSV, fall back to DB queries (original behavior)
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        cur = conn.cursor()
        
        # Convert embedding to pgvector format
        vec_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
        
        # Choose search strategy
        if search_method == "semantic":
            # Pure semantic (vector) search
            query = """
                SELECT 
                    item_id,
                    domain,
                    topic,
                    stem,
                    "choice_A",
                    "choice_B",
                    "choice_C",
                    "choice_D",
                    key,
                    rationale,
                    rasch_b,
                    pvalue,
                    point_biserial,
                    enemy,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM itembank
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(query, (vec_str, vec_str, top_k))
            results = cur.fetchall()
            
            columns = [
                'item_id', 'domain', 'topic', 'stem', 
                'choice_A', 'choice_B', 'choice_C', 'choice_D', 
                'key', 'rationale', 'rasch_b', 'pvalue', 
                'point_biserial', 'enemy', 'similarity_score'
            ]
            
        elif search_method == "hybrid_weighted":
            # Weighted hybrid search (keyword + semantic)
            # Normalize weights
            total_weight = semantic_weight + keyword_weight
            semantic_weight = semantic_weight / total_weight
            keyword_weight = keyword_weight / total_weight
            
            query = """
                WITH 
                keyword_results AS (
                    SELECT 
                        item_id,
                        ts_rank_cd(
                            to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')),
                            plainto_tsquery('english', %s)
                        ) AS keyword_score
                    FROM itembank
                    WHERE to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')) @@ 
                          plainto_tsquery('english', %s)
                ),
                semantic_results AS (
                    SELECT 
                        item_id,
                        1 - (embedding <=> %s::vector) AS semantic_score
                    FROM itembank
                    WHERE embedding IS NOT NULL
                )
                SELECT 
                    i.item_id,
                    i.domain,
                    i.topic,
                    i.stem,
                    i."choice_A",
                    i."choice_B",
                    i."choice_C",
                    i."choice_D",
                    i.key,
                    i.rationale,
                    i.rasch_b,
                    i.pvalue,
                    i.point_biserial,
                    i.enemy,
                    (COALESCE(k.keyword_score, 0) * %s + 
                     COALESCE(s.semantic_score, 0) * %s) AS similarity_score
                FROM itembank i
                LEFT JOIN keyword_results k ON i.item_id = k.item_id
                LEFT JOIN semantic_results s ON i.item_id = s.item_id
                WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
                ORDER BY similarity_score DESC
                LIMIT %s;
            """
            
            cur.execute(query, (keywords, keywords, vec_str, keyword_weight, semantic_weight, top_k))
            results = cur.fetchall()
            
            columns = [
                'item_id', 'domain', 'topic', 'stem', 
                'choice_A', 'choice_B', 'choice_C', 'choice_D', 
                'key', 'rationale', 'rasch_b', 'pvalue', 
                'point_biserial', 'enemy', 'similarity_score'
            ]
            
        elif search_method == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF constant
            
            query = """
                WITH 
                keyword_search AS (
                    SELECT 
                        item_id,
                        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(
                            to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')),
                            plainto_tsquery('english', %s)
                        ) DESC) AS keyword_rank
                    FROM itembank
                    WHERE to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')) @@ 
                          plainto_tsquery('english', %s)
                    LIMIT 50
                ),
                semantic_search AS (
                    SELECT 
                        item_id,
                        ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS semantic_rank
                    FROM itembank
                    WHERE embedding IS NOT NULL
                    LIMIT 50
                )
                SELECT 
                    i.item_id,
                    i.domain,
                    i.topic,
                    i.stem,
                    i."choice_A", i."choice_B", i."choice_C", i."choice_D",
                    i.key,
                    i.rationale,
                    i.rasch_b,
                    i.pvalue,
                    i.point_biserial,
                    i.enemy,
                    (1.0 / (%s + COALESCE(k.keyword_rank, 1000))) +
                    (1.0 / (%s + COALESCE(s.semantic_rank, 1000))) AS similarity_score
                FROM itembank i
                LEFT JOIN keyword_search k ON i.item_id = k.item_id
                LEFT JOIN semantic_search s ON i.item_id = s.item_id
                WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
                ORDER BY similarity_score DESC
                LIMIT %s;
            """
            
            cur.execute(query, (keywords, keywords, vec_str, k, k, top_k))
            results = cur.fetchall()
            
            columns = [
                'item_id', 'domain', 'topic', 'stem', 
                'choice_A', 'choice_B', 'choice_C', 'choice_D', 
                'key', 'rationale', 'rasch_b', 'pvalue', 
                'point_biserial', 'enemy', 'similarity_score'
            ]
        
        else:
            # Default to semantic search
            return vector_search(query_embedding, top_k, "semantic")
        
        df = pd.DataFrame(results, columns=columns)
        
        cur.close()
        conn.close()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        if conn:
            conn.close()
        return pd.DataFrame()

# Helper functions copied from updated main app
def fetch_item_by_id(item_id: str):
    """Fetch a single item row by item_id. Returns (row_tuple, error_str)"""
    # If using CSV, fetch from CSV
    if USE_CSV:
        try:
            df, _ = load_csv_data(CSV_PATH)
            row = df[df['item_id'].astype(str).str.upper() == item_id.strip().upper()]
            if row.empty:
                return None, f"Item {item_id} not found"
            return row_to_item_tuple(row.iloc[0]), None
        except Exception as e:
            return None, str(e)
    
    # Otherwise use database
    conn = get_db_connection()
    if not conn:
        return None, "DB connection failed"
    try:
        cur = conn.cursor()
        cur.execute(
            '''SELECT item_id, domain, topic, stem, "choice_A", "choice_B", "choice_C", "choice_D", key, rationale, rasch_b, pvalue, point_biserial, combined FROM itembank WHERE item_id = %s''',
            (item_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row, None
    except Exception as e:
        return None, str(e)


def render_item_preview(item_row):
    """Render an item preview given a DB row tuple or sequence matching the select order above."""
    if not item_row:
        st.warning("No item to display")
        return
    # support both psycopg2 row tuples and constructed tuples
    try:
        item_id = item_row[0]
        domain = item_row[1]
        topic = item_row[2]
        stem = item_row[3]
        choice_A = item_row[4]
        choice_B = item_row[5]
        choice_C = item_row[6]
        choice_D = item_row[7]
        key = item_row[8]
        rationale = item_row[9]
        rasch_b = item_row[10]
        pvalue = item_row[11]
        point_biserial = item_row[12]
    except Exception:
        st.error("Invalid item format for preview")
        return

    st.markdown(f"<div class=\"metadata\"><strong>🏥 Domain:</strong> {domain}<br><strong>📚 Topic:</strong> {topic}</div>", unsafe_allow_html=True)
    st.markdown('<div class="scenario-title">❓ Question Stem:</div>', unsafe_allow_html=True)
    st.write(stem)
    st.markdown("**📝 Answer Choices:**")
    for choice_letter, choice_text in [('A', choice_A), ('B', choice_B), ('C', choice_C), ('D', choice_D)]:
        is_correct = (choice_letter == key)
        if is_correct:
            st.markdown(f"<div class=\"correct-answer\"><strong>✓ {choice_letter}.</strong> {choice_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class=\"incorrect-answer\"><strong>✗ {choice_letter}.</strong> {choice_text}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style=\"background-color: #fef9e7; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #f39c12;\"><strong>💡 Rationale:</strong><br>{rationale}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class=\"metadata\"><strong>📈 Psychometric Statistics:</strong><br>• <strong>Rasch B (Difficulty):</strong> {rasch_b:.4f}<br>• <strong>P-Value (Proportion Correct):</strong> {pvalue:.4f}<br>• <strong>Point-Biserial (Discrimination):</strong> {point_biserial:.4f}</div>", unsafe_allow_html=True)

# ==================== Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="NCLEX-RN Vector Search (v3)",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #3498db;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .correct-answer {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .incorrect-answer {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .metadata {
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .scenario-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin: 15px 0 10px 0;
        }
        .similarity-score {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔍 NCLEX-RN Vector Database Search (v3)")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Search Configuration")
        
        # Search method selection
        st.subheader("🔍 Search Method")
        search_method = st.selectbox(
            "Choose search strategy:",
            [
                "Pure Semantic (Vector Only)",
                "Weighted Hybrid (Keyword + Semantic)",
                "RRF (Reciprocal Rank Fusion)"
            ],
            help="Select the search algorithm to use"
        )
        
        # Map display names to internal method names
        method_map = {
            "Pure Semantic (Vector Only)": "semantic",
            "Weighted Hybrid (Keyword + Semantic)": "hybrid_weighted",
            "RRF (Reciprocal Rank Fusion)": "rrf"
        }
        search_method_key = method_map[search_method]
        
        # Show method description
        if search_method_key == "semantic":
            st.info("🎯 **Pure Semantic Search**\n\nFinds items based on meaning and context using vector embeddings. Best for conceptual similarity.")
        elif search_method_key == "hybrid_weighted":
            st.info("🎯 **Weighted Hybrid Search**\n\nCombines keyword matching with semantic similarity using adjustable weights.")
        elif search_method_key == "rrf":
            st.info("🎯 **RRF Search**\n\nMerges keyword and semantic rankings using Reciprocal Rank Fusion. No weight tuning needed.")
        
        st.markdown("---")
        
        # Filtering mode selection
        st.subheader("🎚️ Result Filtering")
        filtering_mode = st.radio(
            "Filtering Mode:",
            ["Top K only", "Top P only", "Both Top K and Top P"],
            help="Choose how to filter search results"
        )
        
        # Initialize variables
        top_k = 20  # Default max
        top_p = 0.0  # Default no filter
        
        # Show relevant sliders based on mode
        if filtering_mode in ["Top K only", "Both Top K and Top P"]:
            top_k = st.slider(
                "🔢 Number of similar items (Top K)",
                min_value=1,
                max_value=50,
                value=5,
                help="Select how many similar items to retrieve"
            )
        
        if filtering_mode in ["Top P only", "Both Top K and Top P"]:
            top_p = st.slider(
                "🎯 Minimum Similarity (Top P)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Filter results by minimum similarity score (0 = no filter, 1 = exact match only)"
            )
        
        # Show filtering info
        if filtering_mode == "Top K only":
            st.info(f"📊 Will return top {top_k} results (no similarity threshold)")
        elif filtering_mode == "Top P only":
            st.info(f"📊 Will return all results with similarity ≥ {top_p:.2f}")
        else:
            st.info(f"📊 Will return top {top_k} results with similarity ≥ {top_p:.2f}")
        
        # Additional parameters for hybrid methods
        keywords = None
        semantic_weight = 0.7
        keyword_weight = 0.3
        
        if search_method_key in ["hybrid_weighted", "rrf"]:
            st.markdown("---")
            st.subheader("🔑 Keyword Parameters")
            keywords = st.text_input(
                "Keywords (optional):",
                placeholder="e.g., diabetes insulin glucose",
                help="Enter specific keywords to search for. If empty, uses the query text."
            )
            
            if search_method_key == "hybrid_weighted":
                st.markdown("**Weight Distribution:**")
                semantic_weight = st.slider(
                    "Semantic Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Weight for semantic (vector) similarity"
                )
                keyword_weight = 1.0 - semantic_weight
                st.write(f"Keyword Weight: {keyword_weight:.1f}")
        
        st.markdown("---")
        
        # Database status
        st.subheader("📊 Data Source Status")
        if st.button("🔄 Check Connection"):
            if USE_CSV:
                status, err = csv_status()
                if err:
                    st.error(f"❌ CSV error: {err}")
                else:
                    st.success(
                        f"✅ CSV loaded\n\n"
                        f"📄 Source: {status['path']}\n\n"
                        f"📦 {status['rows']} items\n\n"
                        f"🧠 {status['embedding_count']} items with embeddings"
                    )
            else:
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NOT NULL")
                        count = cur.fetchone()[0]
                        st.success(f"✅ Connected\n\n📦 {count} items with embeddings")
                        cur.close()
                        conn.close()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.error("❌ Connection failed")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Information")
        st.info(f"""
        **Model:** {EMBEDDING_MODEL}  
        **Dimensions:** {EMBEDDING_DIM}  
        **Distance:** Cosine Similarity
        """)
    
    # Main content with tabs
    tab_search, tab_difficulty, tab_classification = st.tabs(["Semantic Similarity Search", "Item Difficulty Prediction", "Item Classification Prediction"])

    # --- Semantic Search Tab (keeps original layout) ---
    with tab_search:
        st.markdown("### Semantic Similarity Search for NCLEX Test Items")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("📝 Input Query")
            input_method = st.radio(
                "Select input method:",
                ["Text Query", "Item ID Lookup"],
                help="Search by text or lookup an existing item by ID",
                key="search_input_method"
            )

            query_text = ""
            if input_method == "Text Query":
                query_text = st.text_area(
                    "Enter your query (stem/scenario):",
                    height=200,
                    placeholder="Example: A patient with diabetes presents with elevated blood glucose levels. Which intervention should the nurse prioritize?",
                    help="Enter a clinical scenario, question stem, or topic to find similar items",
                    key="search_text_query"
                )
            else:
                item_id = st.text_input(
                    "Enter Item ID:",
                    placeholder="NCX0001",
                    help="Enter an existing item ID to find similar items",
                    key="search_item_id"
                )
                if item_id:
                    item, err = fetch_item_by_id(item_id)
                    if err:
                        st.error(f"❌ Error fetching item: {err}")
                    elif item:
                        query_text = item[13]
                        st.markdown('<div class="scenario-title">📋 Selected Item Information</div>', unsafe_allow_html=True)
                        render_item_preview(item)
                    else:
                        st.warning(f"⚠️ Item ID '{item_id}' not found")

            st.markdown("---")
            st.subheader("🎚️ Result Filtering")
            search_clicked = st.button("🔍 Search Similar Items", type="primary")

        output_placeholder = col2.empty()

        if search_clicked:
            if not query_text or not query_text.strip():
                with output_placeholder.container():
                    st.warning("⚠️ Please enter a query or select an item ID")
            else:
                with output_placeholder.container():
                    with st.spinner("🔄 Generating embeddings and searching..."):
                        model = load_embedding_model()
                        query_embedding = generate_embedding(query_text, model)
                        search_keywords = keywords if keywords and keywords.strip() else query_text
                        if filtering_mode == "Top P only":
                            search_top_k = len(load_csv_data(CSV_PATH)[0]) if USE_CSV else 100
                        else:
                            search_top_k = top_k
                        results_df = vector_search(
                            query_embedding,
                            search_top_k,
                            search_method=search_method_key,
                            keywords=search_keywords,
                            semantic_weight=semantic_weight,
                            keyword_weight=keyword_weight
                        )
                        original_results = results_df.copy()
                        if not results_df.empty:
                            results_df = apply_result_filter(results_df, filtering_mode, top_k, top_p)
                        if not results_df.empty:
                            st.success(f"✅ Found {len(results_df)} similar items")
                            st.session_state['original_results'] = original_results
                            st.session_state['filtering_mode'] = filtering_mode
                            st.session_state['top_k'] = top_k
                            st.session_state['top_p'] = top_p
                            st.session_state['search_results'] = results_df
                        else:
                            if filtering_mode in ["Top P only", "Both Top K and Top P"]:
                                st.warning(f"⚠️ No items found with similarity ≥ {top_p:.2f}. Try lowering the Top-P threshold.")
                            else:
                                st.error("❌ No results found")

        if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
            with output_placeholder.container():
                st.subheader("🎯 Similar Items")
                results_df = st.session_state['search_results']

                stored_mode = st.session_state.get('filtering_mode', filtering_mode)
                stored_top_k = st.session_state.get('top_k', top_k)
                stored_top_p = st.session_state.get('top_p', top_p)

                if (stored_mode != filtering_mode or 
                    stored_top_k != top_k or 
                    stored_top_p != top_p):
                    if 'original_results' in st.session_state:
                        results_df = apply_result_filter(
                            st.session_state['original_results'],
                            filtering_mode,
                            top_k,
                            top_p
                        )
                        st.session_state['search_results'] = results_df
                        st.session_state['filtering_mode'] = filtering_mode
                        st.session_state['top_k'] = top_k
                        st.session_state['top_p'] = top_p
                        if results_df.empty:
                            st.warning(f"⚠️ No items found with current filter settings (similarity ≥ {top_p:.2f})")
                            st.stop()

                # Display similar items details table
                st.markdown("**Similar Items Details**")
                detail_cols = ['item_id', 'domain', 'rasch_b', 'enemy', 'similarity_score']
                cols_to_show = [c for c in detail_cols if c in results_df.columns]
                display_results = results_df[cols_to_show].copy()
                rename_dict = {
                    'item_id': 'Item ID',
                    'domain': 'Domain',
                    'rasch_b': 'Rasch B',
                    'enemy': 'Enemy Items',
                    'similarity_score': 'Similarity Score'
                }
                display_results.rename(columns=rename_dict, errors='ignore', inplace=True)
                display_results.index = range(1, len(display_results) + 1)
                st.dataframe(display_results, use_container_width=True)

                if len(results_df) > 1:
                    st.markdown("**Select Item to View:**")
                    selected_idx = st.selectbox(
                        "Choose a similar item:",
                        options=range(len(results_df)),
                        format_func=lambda i: f"Rank #{i+1} - {results_df.iloc[i]['item_id']} (Similarity: {results_df.iloc[i]['similarity_score']:.4f})",
                        key="result_selector_main",
                        label_visibility="collapsed"
                    )
                    items_to_display = [results_df.iloc[selected_idx]]
                    display_rank = selected_idx + 1
                else:
                    items_to_display = [results_df.iloc[0]]
                    display_rank = 1

                for row in items_to_display:
                    st.markdown(f'''
                    <div class="similarity-score">
                        <strong>🔍 Rank #{display_rank} | Item ID: {row['item_id']}</strong><br>
                        📊 Similarity Score: <span style="color: #2980b9; font-size: 1.2em; font-weight: bold;">{row['similarity_score']:.4f}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div class="metadata">
                        <strong>🏥 Domain:</strong> {row['domain']}<br>
                        <strong>📚 Topic:</strong> {row['topic']}
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown(f'<div class="scenario-title">❓ Question Stem:</div>', unsafe_allow_html=True)
                    st.write(row['stem'])
                    st.markdown("**📝 Answer Choices:**")
                    key = row['key']
                    for choice_letter in ['A', 'B', 'C', 'D']:
                        choice_text = row[f'choice_{choice_letter}']
                        is_correct = (choice_letter == key)
                        if is_correct:
                            st.markdown(f'''
                            <div class="correct-answer">
                                <strong>✓ {choice_letter}.</strong> {choice_text}
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="incorrect-answer">
                                <strong>✗ {choice_letter}.</strong> {choice_text}
                            </div>
                            ''', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div style="background-color: #fef9e7; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #f39c12;">
                        <strong>💡 Rationale:</strong><br>
                        {row['rationale']}
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div class="metadata">
                        <strong>📈 Psychometric Statistics:</strong><br>
                        • <strong>Rasch B (Difficulty):</strong> {row['rasch_b']:.4f}<br>
                        • <strong>P-Value (Proportion Correct):</strong> {row['pvalue']:.4f}<br>
                        • <strong>Point-Biserial (Discrimination):</strong> {row['point_biserial']:.4f}
                    </div>
                    ''', unsafe_allow_html=True)
                    if len(results_df) == 1:
                        st.markdown("---")

                st.markdown("---")
                st.subheader("💾 Export Results")
                export_df = results_df[[
                    'item_id', 'similarity_score', 'domain', 'topic', 'stem',
                    'choice_A', 'choice_B', 'choice_C', 'choice_D', 'key',
                    'rasch_b', 'pvalue', 'point_biserial'
                ]]
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="nclex_search_results.csv",
                        mime="text/csv"
                    )
                with col_json:
                    json_data = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="nclex_search_results.json",
                        mime="application/json"
                    )

    # --- Difficulty Prediction Tab ---
    with tab_difficulty:
        st.markdown("### Item Difficulty Prediction")
        dcol1, dcol2 = st.columns([1, 1.5])
        with dcol1:
            st.subheader("📝 Input Query")
            d_input_method = st.radio(
                "Select input method:",
                ["Text Query", "Item ID Lookup"],
                key="diff_input_method",
                help="Search by text or lookup an existing item by ID"
            )
            d_query_text = ""
            if d_input_method == "Text Query":
                d_query_text = st.text_area("Enter your query (stem/scenario):", height=200, key="diff_text_query")
            else:
                d_item_id = st.text_input("Enter Item ID:", key="diff_item_id", placeholder="NCX0001")
                if d_item_id:
                    item, err = fetch_item_by_id(d_item_id)
                    if err:
                        st.error(f"❌ Error fetching item: {err}")
                    elif item:
                        d_query_text = item[13]
                        render_item_preview(item)
                    else:
                        st.warning(f"⚠️ Item ID '{d_item_id}' not found")
            d_search_clicked = st.button("🔍 Predict Difficulty", key="diff_search")

        with dcol2:
            d_output = st.empty()

        if 'difficulty_results' not in st.session_state or st.session_state['difficulty_results'].empty:
            with d_output.container():
                st.info("🔎 Enter a query or item ID, then click Predict Difficulty to compute item difficulty statistics.")

        if d_search_clicked:
            if not d_query_text or not d_query_text.strip():
                with d_output.container():
                    st.warning("⚠️ Please enter a query or select an item ID")
            else:
                with d_output.container():
                    with st.spinner("🔄 Computing difficulty prediction..."):
                        model = load_embedding_model()
                        q_emb = generate_embedding(d_query_text, model)
                        search_keywords = keywords if keywords and keywords.strip() else d_query_text
                        search_top_k = len(load_csv_data(CSV_PATH)[0]) if USE_CSV and filtering_mode == "Top P only" else top_k
                        results_df = vector_search(q_emb, search_top_k, search_method=search_method_key, keywords=search_keywords, semantic_weight=semantic_weight, keyword_weight=keyword_weight)
                        filtered = apply_result_filter(results_df, filtering_mode, top_k, top_p)
                        if filtered.empty:
                            st.warning("⚠️ No similar items found to compute difficulty statistics")
                        else:
                            filtered = filtered.reset_index(drop=True)
                            st.session_state['difficulty_results'] = filtered

        if 'difficulty_results' in st.session_state and not st.session_state['difficulty_results'].empty:
            filtered = st.session_state['difficulty_results']
            with d_output.container():
                rasch_mean = filtered['rasch_b'].mean()
                rasch_sd = filtered['rasch_b'].std(ddof=0)
                st.metric("Average Rasch Difficulty", f"{rasch_mean:.4f}")
                st.metric("Rasch Difficulty SD", f"{rasch_sd:.4f}")
                st.markdown(f"**Items considered:** {len(filtered)}")
                stats_df = filtered[[ 'item_id', 'domain', 'rasch_b', 'similarity_score' ]].copy()
                stats_df.index = range(1, len(stats_df) + 1)
                st.dataframe(stats_df, use_container_width=True)
                st.markdown("---")
                st.markdown("**Select Item to View:**")
                sel_idx = st.selectbox(
                    "Choose an item:",
                    options=range(len(filtered)),
                    format_func=lambda i: f"{filtered.iloc[i]['item_id']} (Rasch B: {filtered.iloc[i]['rasch_b']:.3f})",
                    key="difficulty_select_idx"
                )
                sel_row = filtered.iloc[sel_idx]
                render_item_preview((sel_row['item_id'], sel_row['domain'], sel_row['topic'], sel_row['stem'], sel_row['choice_A'], sel_row['choice_B'], sel_row['choice_C'], sel_row['choice_D'], sel_row['key'], sel_row['rationale'], sel_row['rasch_b'], sel_row['pvalue'], sel_row['point_biserial']))

    # --- Classification Prediction Tab ---
    with tab_classification:
        st.markdown("### Item Classification Prediction")
        ccol1, ccol2 = st.columns([1, 1.5])
        with ccol1:
            st.subheader("📝 Input Query")
            c_input_method = st.radio(
                "Select input method:",
                ["Text Query", "Item ID Lookup"],
                key="class_input_method",
                help="Search by text or lookup an existing item by ID"
            )
            c_query_text = ""
            if c_input_method == "Text Query":
                c_query_text = st.text_area("Enter your query (stem/scenario):", height=200, key="class_text_query")
            else:
                c_item_id = st.text_input("Enter Item ID:", key="class_item_id", placeholder="NCX0001")
                if c_item_id:
                    item, err = fetch_item_by_id(c_item_id)
                    if err:
                        st.error(f"❌ Error fetching item: {err}")
                    elif item:
                        c_query_text = item[13]
                        render_item_preview(item)
                    else:
                        st.warning(f"⚠️ Item ID '{c_item_id}' not found")
            c_search_clicked = st.button("🔍 Predict Classification", key="class_search")

        with ccol2:
            c_output = st.empty()

        if 'classification_results' not in st.session_state or st.session_state['classification_results'].empty:
            with c_output.container():
                st.info("🔎 Enter a query or item ID, then click Predict Classification to view the predicted domain and item breakdown.")

        if c_search_clicked:
            if not c_query_text or not c_query_text.strip():
                with c_output.container():
                    st.warning("⚠️ Please enter a query or select an item ID")
            else:
                with c_output.container():
                    with st.spinner("🔄 Generating classification results..."):
                        model = load_embedding_model()
                        q_emb = generate_embedding(c_query_text, model)
                        search_keywords = keywords if keywords and keywords.strip() else c_query_text
                        search_top_k = len(load_csv_data(CSV_PATH)[0]) if USE_CSV and filtering_mode == "Top P only" else top_k
                        results_df = vector_search(
                            q_emb,
                            search_top_k,
                            search_method=search_method_key,
                            keywords=search_keywords,
                            semantic_weight=semantic_weight,
                            keyword_weight=keyword_weight
                        )
                        filtered = apply_result_filter(results_df, filtering_mode, top_k, top_p)
                        st.session_state['classification_original_results'] = results_df.copy()
                        st.session_state['classification_filtering_mode'] = filtering_mode
                        st.session_state['classification_top_k'] = top_k
                        st.session_state['classification_top_p'] = top_p
                        if filtered.empty:
                            st.session_state['classification_results'] = pd.DataFrame()
                            st.warning("⚠️ No similar items found to compute classification")
                        else:
                            st.session_state['classification_results'] = filtered

        if 'classification_original_results' in st.session_state:
            stored_class_mode = st.session_state.get('classification_filtering_mode', filtering_mode)
            stored_class_top_k = st.session_state.get('classification_top_k', top_k)
            stored_class_top_p = st.session_state.get('classification_top_p', top_p)

            if (
                stored_class_mode != filtering_mode or
                stored_class_top_k != top_k or
                stored_class_top_p != top_p
            ):
                filtered = apply_result_filter(
                    st.session_state['classification_original_results'],
                    filtering_mode,
                    top_k,
                    top_p
                )
                st.session_state['classification_results'] = filtered
                st.session_state['classification_filtering_mode'] = filtering_mode
                st.session_state['classification_top_k'] = top_k
                st.session_state['classification_top_p'] = top_p

        if 'classification_results' in st.session_state and not st.session_state['classification_results'].empty:
            filtered = st.session_state['classification_results']
            with c_output.container():
                domain_counts = (
                    filtered
                    .groupby('domain', as_index=False)
                    .agg(
                        count=('domain', 'size'),
                        mean_similarity_score=('similarity_score', 'mean')
                    )
                    .sort_values(
                        ['count', 'mean_similarity_score'],
                        ascending=[False, False]
                    )
                    .reset_index(drop=True)
                )
                domain_counts.index = range(1, len(domain_counts) + 1)
                predicted_domain = domain_counts.iloc[0]['domain']
                st.metric("Predicted Domain", predicted_domain)
                st.markdown("**Domain Summary for Similar Items**")
                st.dataframe(domain_counts, use_container_width=True)

                st.markdown("**Similar Items Details**")
                detail_cols = ['item_id', 'domain', 'rasch_b', 'enemy', 'similarity_score']
                cols_to_show = [c for c in detail_cols if c in filtered.columns]
                display_filtered = filtered[cols_to_show].copy()
                rename_dict = {
                    'item_id': 'Item ID',
                    'domain': 'Domain',
                    'rasch_b': 'Rasch B',
                    'enemy': 'Enemy Items',
                    'similarity_score': 'Similarity Score'
                }
                display_filtered.rename(columns=rename_dict, errors='ignore', inplace=True)
                display_filtered.index = range(1, len(display_filtered) + 1)
                st.dataframe(display_filtered, use_container_width=True)

                st.markdown("---")
                st.markdown("**Select Item to View:**")
                sel_idx = st.selectbox(
                    "Choose an item:",
                    options=range(len(filtered)),
                    format_func=lambda i: f"{filtered.iloc[i]['item_id']} ({filtered.iloc[i]['domain']})",
                    key="classification_select_idx"
                )
                sel_row = filtered.iloc[sel_idx]
                render_item_preview((sel_row['item_id'], sel_row['domain'], sel_row['topic'], sel_row['stem'], sel_row['choice_A'], sel_row['choice_B'], sel_row['choice_C'], sel_row['choice_D'], sel_row['key'], sel_row['rationale'], sel_row['rasch_b'], sel_row['pvalue'], sel_row['point_biserial']))
        elif 'classification_original_results' in st.session_state:
            with c_output.container():
                st.warning(f"⚠️ No similar items found with current filter settings (similarity ≥ {top_p:.2f})")

if __name__ == "__main__":
    main()
