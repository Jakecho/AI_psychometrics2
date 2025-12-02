"""
Hybrid Search Implementation for NCLEX Item Bank
Combines Keyword (Full-Text) + Semantic (Vector) Search
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Tuple

# Configuration
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded!")

def create_fulltext_index():
    """Create PostgreSQL full-text search index if not exists"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
        # Create full-text search index on stem + combined text
        cur.execute("""
            CREATE INDEX IF NOT EXISTS itembank_fts_idx 
            ON itembank USING GIN (to_tsvector('english', 
                                    COALESCE(stem, '') || ' ' || 
                                    COALESCE(combined, '')));
        """)
        conn.commit()
        print("✅ Full-text search index created/verified")
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def hybrid_search_weighted(
    query_text: str,
    keywords: str = None,
    top_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    min_keyword_score: float = 0.0
) -> pd.DataFrame:
    """
    Hybrid search combining keyword and semantic similarity
    
    Args:
        query_text: Text to generate embedding from
        keywords: Optional specific keywords to search (if None, uses query_text)
        top_k: Number of results to return
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        min_keyword_score: Minimum keyword score threshold
        
    Returns:
        DataFrame with results and scores
    """
    
    if keywords is None:
        keywords = query_text
    
    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    semantic_weight = semantic_weight / total_weight
    keyword_weight = keyword_weight / total_weight
    
    # Generate query embedding
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
        # Hybrid search query
        query = """
            WITH 
            -- Keyword search using full-text search
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
                  AND ts_rank_cd(
                        to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')),
                        plainto_tsquery('english', %s)
                      ) > %s
            ),
            -- Semantic search using vector similarity
            semantic_results AS (
                SELECT 
                    item_id,
                    1 - (embedding <=> %s::vector) AS semantic_score
                FROM itembank
                WHERE embedding IS NOT NULL
            )
            -- Combine results with weighted scoring
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
                COALESCE(k.keyword_score, 0) AS keyword_score,
                COALESCE(s.semantic_score, 0) AS semantic_score,
                (COALESCE(k.keyword_score, 0) * %s + 
                 COALESCE(s.semantic_score, 0) * %s) AS hybrid_score
            FROM itembank i
            LEFT JOIN keyword_results k ON i.item_id = k.item_id
            LEFT JOIN semantic_results s ON i.item_id = s.item_id
            WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
            ORDER BY hybrid_score DESC
            LIMIT %s;
        """
        
        cur.execute(query, (
            keywords, keywords, keywords, min_keyword_score,  # keyword search params
            vec_str,  # semantic search param
            keyword_weight, semantic_weight,  # weights
            top_k  # limit
        ))
        
        results = cur.fetchall()
        
        columns = [
            'item_id', 'domain', 'topic', 'stem',
            'choice_A', 'choice_B', 'choice_C', 'choice_D',
            'key', 'rationale', 'rasch_b', 'pvalue', 'point_biserial',
            'keyword_score', 'semantic_score', 'hybrid_score'
        ]
        
        df = pd.DataFrame(results, columns=columns)
        return df
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return pd.DataFrame()
    finally:
        cur.close()
        conn.close()

def pure_keyword_search(keywords: str, top_k: int = 10) -> pd.DataFrame:
    """Pure keyword-based search using PostgreSQL full-text search"""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
        query = """
            SELECT 
                item_id,
                domain,
                topic,
                stem,
                "choice_A", "choice_B", "choice_C", "choice_D",
                key,
                rationale,
                rasch_b,
                pvalue,
                point_biserial,
                ts_rank_cd(
                    to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')),
                    plainto_tsquery('english', %s)
                ) AS keyword_score
            FROM itembank
            WHERE to_tsvector('english', COALESCE(stem, '') || ' ' || COALESCE(combined, '')) @@ 
                  plainto_tsquery('english', %s)
            ORDER BY keyword_score DESC
            LIMIT %s;
        """
        
        cur.execute(query, (keywords, keywords, top_k))
        results = cur.fetchall()
        
        columns = [
            'item_id', 'domain', 'topic', 'stem',
            'choice_A', 'choice_B', 'choice_C', 'choice_D',
            'key', 'rationale', 'rasch_b', 'pvalue', 'point_biserial',
            'keyword_score'
        ]
        
        df = pd.DataFrame(results, columns=columns)
        return df
        
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
        return pd.DataFrame()
    finally:
        cur.close()
        conn.close()

def rrf_hybrid_search(query_text: str, keywords: str = None, top_k: int = 10, k: int = 60) -> pd.DataFrame:
    """
    Reciprocal Rank Fusion (RRF) hybrid search
    Combines rankings from keyword and semantic search
    
    Args:
        query_text: Text for semantic embedding
        keywords: Keywords for full-text search
        top_k: Final number of results
        k: RRF constant (typically 60)
    """
    
    if keywords is None:
        keywords = query_text
    
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
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
                k.keyword_rank,
                s.semantic_rank,
                -- RRF Score: 1/(k + rank)
                (1.0 / (%s + COALESCE(k.keyword_rank, 1000))) +
                (1.0 / (%s + COALESCE(s.semantic_rank, 1000))) AS rrf_score
            FROM itembank i
            LEFT JOIN keyword_search k ON i.item_id = k.item_id
            LEFT JOIN semantic_search s ON i.item_id = s.item_id
            WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
            ORDER BY rrf_score DESC
            LIMIT %s;
        """
        
        cur.execute(query, (keywords, keywords, vec_str, k, k, top_k))
        results = cur.fetchall()
        
        columns = [
            'item_id', 'domain', 'topic', 'stem',
            'choice_A', 'choice_B', 'choice_C', 'choice_D',
            'key', 'rationale', 'rasch_b', 'pvalue', 'point_biserial',
            'keyword_rank', 'semantic_rank', 'rrf_score'
        ]
        
        df = pd.DataFrame(results, columns=columns)
        return df
        
    except Exception as e:
        print(f"❌ RRF search failed: {e}")
        return pd.DataFrame()
    finally:
        cur.close()
        conn.close()

# ============= DEMO USAGE =============

if __name__ == "__main__":
    # Create index first time
    create_fulltext_index()
    
    print("\n" + "="*80)
    print("🔍 DEMO: Hybrid Search for NCLEX Items")
    print("="*80)
    
    # Example query
    query = "diabetes patient with high blood sugar needs insulin"
    keywords = "diabetes insulin glucose"
    
    print(f"\n📝 Query: {query}")
    print(f"🔑 Keywords: {keywords}\n")
    
    # 1. Pure keyword search
    print("\n1️⃣ PURE KEYWORD SEARCH")
    print("-" * 80)
    keyword_results = pure_keyword_search(keywords, top_k=5)
    if not keyword_results.empty:
        for idx, row in keyword_results.iterrows():
            print(f"\nRank {idx+1}: {row['item_id']} (Score: {row['keyword_score']:.4f})")
            print(f"  Domain: {row['domain']}")
            print(f"  Stem: {row['stem'][:100]}...")
    
    # 2. Weighted hybrid search (70% semantic, 30% keyword)
    print("\n\n2️⃣ WEIGHTED HYBRID SEARCH (70% Semantic + 30% Keyword)")
    print("-" * 80)
    hybrid_results = hybrid_search_weighted(
        query_text=query,
        keywords=keywords,
        top_k=5,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    if not hybrid_results.empty:
        for idx, row in hybrid_results.iterrows():
            print(f"\nRank {idx+1}: {row['item_id']} (Hybrid: {row['hybrid_score']:.4f})")
            print(f"  Keyword: {row['keyword_score']:.4f} | Semantic: {row['semantic_score']:.4f}")
            print(f"  Domain: {row['domain']}")
            print(f"  Stem: {row['stem'][:100]}...")
    
    # 3. RRF hybrid search
    print("\n\n3️⃣ RRF (RECIPROCAL RANK FUSION)")
    print("-" * 80)
    rrf_results = rrf_hybrid_search(query, keywords, top_k=5)
    if not rrf_results.empty:
        for idx, row in rrf_results.iterrows():
            print(f"\nRank {idx+1}: {row['item_id']} (RRF: {row['rrf_score']:.4f})")
            print(f"  Keyword Rank: {row['keyword_rank'] if pd.notna(row['keyword_rank']) else 'N/A'} | "
                  f"Semantic Rank: {row['semantic_rank'] if pd.notna(row['semantic_rank']) else 'N/A'}")
            print(f"  Domain: {row['domain']}")
            print(f"  Stem: {row['stem'][:100]}...")
    
    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
