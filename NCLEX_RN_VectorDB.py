"""
NCLEX-RN Vector Database Search System
Performs semantic similarity search on NCLEX item bank using pgvector
"""

import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from typing import List, Dict, Any
import numpy as np

# ==================== Configuration ====================
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ==================== Helper Functions ====================

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    return SentenceTransformer(EMBEDDING_MODEL)

def get_db_connection():
    """Create database connection"""
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
    Perform vector similarity search in pgvector database with multiple search strategies
    
    Args:
        query_embedding: The embedding vector of the query
        top_k: Number of similar items to return
        search_method: Search strategy - "semantic", "hybrid_weighted", or "rrf"
        keywords: Keywords for keyword-based search (used in hybrid methods)
        semantic_weight: Weight for semantic similarity (hybrid_weighted only)
        keyword_weight: Weight for keyword matching (hybrid_weighted only)
        
    Returns:
        DataFrame with similar items and their metadata
    """
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
                'point_biserial', 'similarity_score'
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
                'point_biserial', 'similarity_score'
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
                'point_biserial', 'similarity_score'
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

# ==================== Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="NCLEX-RN Vector Search",
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
    st.title("🔍 NCLEX-RN Vector Database Search")
    st.markdown("### Semantic Similarity Search for NCLEX Test Items")
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
        st.subheader("📊 Database Status")
        if st.button("🔄 Check Connection"):
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
    
    # Main content
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📝 Input Query")
        
        # Query input method selection
        input_method = st.radio(
            "Select input method:",
            ["Text Query", "Item ID Lookup"],
            help="Search by text or lookup an existing item by ID"
        )
        
        query_text = ""
        
        if input_method == "Text Query":
            query_text = st.text_area(
                "Enter your query (stem/scenario):",
                height=200,
                placeholder="Example: A patient with diabetes presents with elevated blood glucose levels. Which intervention should the nurse prioritize?",
                help="Enter a clinical scenario, question stem, or topic to find similar items"
            )
        else:
            # Item ID lookup
            item_id = st.text_input(
                "Enter Item ID:",
                placeholder="NCX0001",
                help="Enter an existing item ID to find similar items"
            )
            
            if item_id:
                # Fetch the item from database
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            '''SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
                               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
                               point_biserial, combined 
                               FROM itembank WHERE item_id = %s''',
                            (item_id,)
                        )
                        result = cur.fetchone()
                        if result:
                            query_text = result[13]  # Use combined text
                            
                            # Display full item information
                            st.markdown('<div class="scenario-title">📋 Selected Item Information</div>', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="metadata">
                                <strong>🏥 Domain:</strong> {result[1]}<br>
                                <strong>📚 Topic:</strong> {result[2]}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown('<div class="scenario-title">❓ Question Stem:</div>', unsafe_allow_html=True)
                            st.write(result[3])
                            
                            st.markdown("**📝 Answer Choices:**")
                            key = result[8]
                            for choice_letter, choice_idx in [('A', 4), ('B', 5), ('C', 6), ('D', 7)]:
                                is_correct = (choice_letter == key)
                                if is_correct:
                                    st.markdown(f'''
                                    <div class="correct-answer">
                                        <strong>✓ {choice_letter}.</strong> {result[choice_idx]}
                                    </div>
                                    ''', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'''
                                    <div class="incorrect-answer">
                                        <strong>✗ {choice_letter}.</strong> {result[choice_idx]}
                                    </div>
                                    ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div style="background-color: #fef9e7; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #f39c12;">
                                <strong>💡 Rationale:</strong><br>
                                {result[9]}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="metadata">
                                <strong>📈 Psychometric Statistics:</strong><br>
                                • <strong>Rasch B:</strong> {result[10]:.4f}<br>
                                • <strong>P-Value:</strong> {result[11]:.4f}<br>
                                • <strong>Point-Biserial:</strong> {result[12]:.4f}
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.warning(f"⚠️ Item ID '{item_id}' not found")
                        cur.close()
                        conn.close()
                    except Exception as e:
                        st.error(f"❌ Error fetching item: {str(e)}")
        
        # Search button
        search_clicked = st.button("🔍 Search Similar Items", type="primary")
    
    # Create placeholder for output in col2
    with col2:
        output_placeholder = st.empty()
    
    # Handle search in col1 but display results in col2 placeholder
    if search_clicked:
        if not query_text or not query_text.strip():
            with output_placeholder.container():
                st.warning("⚠️ Please enter a query or select an item ID")
        else:
            with output_placeholder.container():
                with st.spinner("🔄 Generating embeddings and searching..."):
                    # Load model
                    model = load_embedding_model()
                    
                    # Generate embedding
                    query_embedding = generate_embedding(query_text, model)
                    
                    # Use keywords if provided, otherwise use query text
                    search_keywords = keywords if keywords and keywords.strip() else query_text
                    
                    # Determine search parameters based on filtering mode
                    if filtering_mode == "Top P only":
                        # Fetch more results initially, then filter by Top P
                        search_top_k = 100  # Get more results for Top P filtering
                    else:
                        search_top_k = top_k
                    
                    # Perform search with selected method
                    results_df = vector_search(
                        query_embedding, 
                        search_top_k, 
                        search_method=search_method_key,
                        keywords=search_keywords,
                        semantic_weight=semantic_weight,
                        keyword_weight=keyword_weight
                    )
                    
                    # Apply filtering based on mode
                    original_results = results_df.copy()  # Store original results
                    
                    if not results_df.empty:
                        if filtering_mode == "Top P only":
                            # Filter by similarity threshold only
                            results_df = results_df[results_df['similarity_score'] >= top_p]
                        elif filtering_mode == "Both Top K and Top P":
                            # Apply both filters: Top K first, then Top P
                            results_df = results_df.head(top_k)
                            results_df = results_df[results_df['similarity_score'] >= top_p]
                        # For "Top K only", results are already limited by search_top_k
                    
                    if not results_df.empty:
                        st.success(f"✅ Found {len(results_df)} similar items")
                        
                        # Store both original and filtered results in session state
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
    
    # Display results in col2
    if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
        with output_placeholder.container():
            st.subheader("🎯 Similar Items")
            
            results_df = st.session_state['search_results']
            
            # Re-apply filtering if settings have changed
            stored_mode = st.session_state.get('filtering_mode', filtering_mode)
            stored_top_k = st.session_state.get('top_k', top_k)
            stored_top_p = st.session_state.get('top_p', top_p)
            
            # Check if filtering parameters have changed
            if (stored_mode != filtering_mode or 
                stored_top_k != top_k or 
                stored_top_p != top_p):
                
                # Get original unfiltered results if available
                if 'original_results' in st.session_state:
                    results_df = st.session_state['original_results'].copy()
                    
                    # Apply current filtering
                    if filtering_mode == "Top P only":
                        results_df = results_df[results_df['similarity_score'] >= top_p]
                    elif filtering_mode == "Both Top K and Top P":
                        results_df = results_df.head(top_k)
                        results_df = results_df[results_df['similarity_score'] >= top_p]
                    elif filtering_mode == "Top K only":
                        results_df = results_df.head(top_k)
                    
                    # Update session state
                    st.session_state['search_results'] = results_df
                    st.session_state['filtering_mode'] = filtering_mode
                    st.session_state['top_k'] = top_k
                    st.session_state['top_p'] = top_p
                    
                    if results_df.empty:
                        st.warning(f"⚠️ No items found with current filter settings (similarity ≥ {top_p:.2f})")
                        st.stop()
            
            # Add dropdown selector if multiple results
            if len(results_df) > 1:
                st.markdown("**Select Item to View:**")
                selected_idx = st.selectbox(
                    "Choose a similar item:",
                    options=range(len(results_df)),
                    format_func=lambda i: f"Rank #{i+1} - {results_df.iloc[i]['item_id']} (Similarity: {results_df.iloc[i]['similarity_score']:.4f})",
                    key="result_selector",
                    label_visibility="collapsed"
                )
                items_to_display = [results_df.iloc[selected_idx]]
                display_rank = selected_idx + 1
            else:
                items_to_display = [results_df.iloc[0]]
                display_rank = 1
            
            # Display selected item(s)
            for row in items_to_display:
                # Similarity Score Banner
                st.markdown(f'''
                <div class="similarity-score">
                    <strong>🔍 Rank #{display_rank} | Item ID: {row['item_id']}</strong><br>
                    📊 Similarity Score: <span style="color: #2980b9; font-size: 1.2em; font-weight: bold;">{row['similarity_score']:.4f}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                # Domain and Topic
                st.markdown(f'''
                <div class="metadata">
                    <strong>🏥 Domain:</strong> {row['domain']}<br>
                    <strong>📚 Topic:</strong> {row['topic']}
                </div>
                ''', unsafe_allow_html=True)
                
                # Stem/Question
                st.markdown(f'<div class="scenario-title">❓ Question Stem:</div>', unsafe_allow_html=True)
                st.write(row['stem'])
                
                # Answer Choices
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
                
                # Rationale
                st.markdown(f'''
                <div style="background-color: #fef9e7; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #f39c12;">
                    <strong>💡 Rationale:</strong><br>
                    {row['rationale']}
                </div>
                ''', unsafe_allow_html=True)
                
                # Psychometric Stats
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
            
            # Download option
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            # Prepare export data
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

if __name__ == "__main__":
    main()
