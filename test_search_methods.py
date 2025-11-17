"""
Quick test of all three search methods
"""

import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test query
query_text = "patient with high blood pressure needs medication"
keywords = "hypertension medication blood pressure"

print(f"\n{'='*80}")
print(f"Test Query: {query_text}")
print(f"Keywords: {keywords}")
print(f"{'='*80}\n")

# Generate embedding
embedding = model.encode([query_text], normalize_embeddings=True)[0]
vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Test 1: Pure Semantic
print("1️⃣ PURE SEMANTIC SEARCH")
print("-" * 80)
query = """
    SELECT DISTINCT ON (item_id)
        item_id,
        domain,
        1 - (embedding <=> %s::vector) AS similarity_score
    FROM itembank
    WHERE embedding IS NOT NULL
    ORDER BY item_id, embedding <=> %s::vector
    LIMIT 5
"""
cur.execute(query, (vec_str, vec_str))
results = cur.fetchall()
for i, row in enumerate(results, 1):
    print(f"{i}. {row[0]} | {row[1]} | Score: {row[2]:.4f}")

# Test 2: Weighted Hybrid
print(f"\n2️⃣ WEIGHTED HYBRID SEARCH (70% Semantic + 30% Keyword)")
print("-" * 80)
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
        COALESCE(k.keyword_score, 0) AS kw_score,
        COALESCE(s.semantic_score, 0) AS sem_score,
        (COALESCE(k.keyword_score, 0) * 0.3 + 
         COALESCE(s.semantic_score, 0) * 0.7) AS hybrid_score
    FROM itembank i
    LEFT JOIN keyword_results k ON i.item_id = k.item_id
    LEFT JOIN semantic_results s ON i.item_id = s.item_id
    WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
    ORDER BY hybrid_score DESC
    LIMIT 5;
"""
cur.execute(query, (keywords, keywords, vec_str))
results = cur.fetchall()
for i, row in enumerate(results, 1):
    print(f"{i}. {row[0]} | {row[1]}")
    print(f"   KW: {row[2]:.4f} | Sem: {row[3]:.4f} | Hybrid: {row[4]:.4f}")

# Test 3: RRF
print(f"\n3️⃣ RRF (RECIPROCAL RANK FUSION)")
print("-" * 80)
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
        k.keyword_rank,
        s.semantic_rank,
        (1.0 / (60 + COALESCE(k.keyword_rank, 1000))) +
        (1.0 / (60 + COALESCE(s.semantic_rank, 1000))) AS rrf_score
    FROM itembank i
    LEFT JOIN keyword_search k ON i.item_id = k.item_id
    LEFT JOIN semantic_search s ON i.item_id = s.item_id
    WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
    ORDER BY rrf_score DESC
    LIMIT 5;
"""
cur.execute(query, (keywords, keywords, vec_str))
results = cur.fetchall()
for i, row in enumerate(results, 1):
    kw_rank = row[2] if row[2] else "N/A"
    sem_rank = row[3] if row[3] else "N/A"
    print(f"{i}. {row[0]} | {row[1]}")
    print(f"   KW Rank: {kw_rank} | Sem Rank: {sem_rank} | RRF: {row[4]:.4f}")

cur.close()
conn.close()

print(f"\n{'='*80}")
print("✅ All search methods tested successfully!")
print(f"{'='*80}")
print("\n🌐 Open http://localhost:8501 to test in the web interface")
