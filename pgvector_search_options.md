# pgvector Search Options & Hybrid Search Strategies

## Overview of pgvector Search Capabilities

### 1. **Vector Distance Operators**

pgvector provides three distance operators for similarity search:

| Operator | Distance Type | Use Case | Formula |
|----------|--------------|----------|---------|
| `<->` | **L2 Distance (Euclidean)** | General similarity | √(Σ(a-b)²) |
| `<#>` | **Inner Product (Negative)** | When vectors not normalized | -(a·b) |
| `<=>` | **Cosine Distance** | **Best for semantic similarity** | 1 - (a·b)/(‖a‖‖b‖) |

**Current Implementation:** We use `<=>` (Cosine Distance) - optimal for semantic similarity!

```sql
-- Cosine Distance (RECOMMENDED for semantic search)
SELECT * FROM itembank 
ORDER BY embedding <=> '[0.1,0.2,...]'::vector 
LIMIT 5;

-- L2 Distance
SELECT * FROM itembank 
ORDER BY embedding <-> '[0.1,0.2,...]'::vector 
LIMIT 5;

-- Inner Product
SELECT * FROM itembank 
ORDER BY embedding <#> '[0.1,0.2,...]'::vector 
LIMIT 5;
```

---

## 2. **Hybrid Search: Combining Keyword + Semantic Search**

### **Option A: PostgreSQL Full-Text Search + Vector Search**

Combine traditional keyword matching with semantic similarity:

```sql
-- Create full-text search index
CREATE INDEX IF NOT EXISTS itembank_fts_idx 
ON itembank USING GIN (to_tsvector('english', stem || ' ' || combined));

-- Hybrid Search: Keyword + Semantic
SELECT 
    item_id,
    domain,
    topic,
    stem,
    -- Keyword match score
    ts_rank(to_tsvector('english', stem || ' ' || combined), 
            plainto_tsquery('english', 'diabetes insulin')) AS keyword_score,
    -- Semantic similarity score
    1 - (embedding <=> '[...]'::vector) AS semantic_score,
    -- Combined score (weighted)
    (0.3 * ts_rank(to_tsvector('english', stem || ' ' || combined), 
                   plainto_tsquery('english', 'diabetes insulin'))) +
    (0.7 * (1 - (embedding <=> '[...]'::vector))) AS hybrid_score
FROM itembank
WHERE 
    to_tsvector('english', stem || ' ' || combined) @@ 
    plainto_tsquery('english', 'diabetes insulin')
    AND embedding IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;
```

**Weights:**
- `0.3` = Keyword importance (exact term matching)
- `0.7` = Semantic importance (meaning/topic matching)
- Adjust these based on your needs!

---

### **Option B: BM25 + Vector Search (Advanced)**

BM25 is superior to TF-IDF for keyword ranking:

```sql
-- Using pg_trgm for fuzzy matching + vector
CREATE EXTENSION IF NOT EXISTS pg_trgm;

SELECT 
    item_id,
    stem,
    -- Trigram similarity for fuzzy keyword matching
    similarity(stem, 'diabetic patient insulin') AS keyword_sim,
    -- Semantic similarity
    1 - (embedding <=> '[...]'::vector) AS semantic_sim,
    -- Hybrid score
    (0.4 * similarity(stem, 'diabetic patient insulin')) +
    (0.6 * (1 - (embedding <=> '[...]'::vector))) AS hybrid_score
FROM itembank
WHERE similarity(stem, 'diabetic patient insulin') > 0.1
ORDER BY hybrid_score DESC
LIMIT 10;
```

---

### **Option C: RRF (Reciprocal Rank Fusion)**

Combine rankings from multiple search methods:

```sql
WITH keyword_search AS (
    SELECT 
        item_id,
        ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) AS keyword_rank
    FROM itembank
    WHERE to_tsvector(...) @@ plainto_tsquery(...)
    LIMIT 20
),
semantic_search AS (
    SELECT 
        item_id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> '[...]'::vector) AS semantic_rank
    FROM itembank
    WHERE embedding IS NOT NULL
    LIMIT 20
)
SELECT 
    COALESCE(k.item_id, s.item_id) AS item_id,
    -- RRF Score: 1/(k + rank_i) where k=60 is standard
    (1.0 / (60 + COALESCE(k.keyword_rank, 1000))) +
    (1.0 / (60 + COALESCE(s.semantic_rank, 1000))) AS rrf_score
FROM keyword_search k
FULL OUTER JOIN semantic_search s ON k.item_id = s.item_id
ORDER BY rrf_score DESC
LIMIT 10;
```

---

## 3. **Filtered Vector Search**

Add WHERE clauses to filter by metadata before semantic search:

```sql
-- Filter by domain + semantic search
SELECT item_id, stem, 1 - (embedding <=> '[...]'::vector) AS similarity
FROM itembank
WHERE domain = 'Pharmacological & Parenteral Therapies'
  AND embedding IS NOT NULL
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;

-- Filter by difficulty range + semantic search
SELECT item_id, stem, rasch_b, pvalue
FROM itembank
WHERE rasch_b BETWEEN -1.0 AND 1.0  -- Medium difficulty
  AND pvalue BETWEEN 0.3 AND 0.7
  AND embedding IS NOT NULL
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;

-- Multi-criteria filtering
SELECT item_id, domain, topic, stem
FROM itembank
WHERE domain IN ('Basic Care & Comfort', 'Safety & Infection Control')
  AND point_biserial > 0.2  -- Good discrimination
  AND embedding IS NOT NULL
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;
```

---

## 4. **Topic-Based Clustering Search**

Find items in the same semantic cluster:

```sql
-- Find items semantically similar to a reference item
WITH reference_item AS (
    SELECT embedding 
    FROM itembank 
    WHERE item_id = 'NCX0001'
)
SELECT 
    i.item_id,
    i.domain,
    i.topic,
    1 - (i.embedding <=> r.embedding) AS similarity
FROM itembank i, reference_item r
WHERE i.embedding IS NOT NULL
  AND i.item_id != 'NCX0001'
ORDER BY i.embedding <=> r.embedding
LIMIT 20;
```

---

## 5. **IVFFLAT & HNSW Indexes for Performance**

For large datasets (>10k items), use approximate nearest neighbor indexes:

```sql
-- Create IVFFLAT index (faster, slightly less accurate)
CREATE INDEX ON itembank 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- √(row_count) is recommended

-- Or HNSW index (slower build, faster search, more accurate)
CREATE INDEX ON itembank 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Performance tuning:**
```sql
-- For IVFFLAT: Set probes (higher = more accurate, slower)
SET ivfflat.probes = 10;  -- Default is 1

-- For HNSW: Set ef_search (higher = more accurate, slower)
SET hnsw.ef_search = 40;  -- Default is 40
```

---

## 6. **Multi-Vector Search (Multiple Embeddings)**

Search using multiple query embeddings simultaneously:

```sql
-- Average multiple embeddings for better coverage
WITH query_vectors AS (
    SELECT 
        '[0.1,0.2,...]'::vector AS emb1,  -- "diabetes"
        '[0.3,0.4,...]'::vector AS emb2   -- "insulin management"
)
SELECT 
    i.item_id,
    i.stem,
    ((1 - (i.embedding <=> q.emb1)) + 
     (1 - (i.embedding <=> q.emb2))) / 2 AS avg_similarity
FROM itembank i, query_vectors q
WHERE i.embedding IS NOT NULL
ORDER BY avg_similarity DESC
LIMIT 10;
```

---

## 7. **Diversity Search (Maximum Marginal Relevance)**

Avoid redundant results by maximizing diversity:

```python
# Python implementation
def mmr_search(query_embedding, items, lambda_param=0.5):
    """
    Maximum Marginal Relevance
    lambda_param: 1.0 = only relevance, 0.0 = only diversity
    """
    selected = []
    candidates = items.copy()
    
    while len(selected) < top_k and candidates:
        mmr_scores = []
        for item in candidates:
            relevance = cosine_similarity(query_embedding, item.embedding)
            if not selected:
                diversity = 0
            else:
                diversity = max(
                    cosine_similarity(item.embedding, s.embedding) 
                    for s in selected
                )
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr, item))
        
        mmr_scores.sort(reverse=True)
        selected.append(mmr_scores[0][1])
        candidates.remove(mmr_scores[0][1])
    
    return selected
```

---

## 8. **Recommended Hybrid Strategy for NCLEX Item Bank**

### **Best Practice: Weighted Hybrid Search**

```sql
-- Recommended query combining all techniques
WITH 
-- 1. Exact keyword matches
keyword_results AS (
    SELECT 
        item_id,
        ts_rank_cd(to_tsvector('english', stem || ' ' || combined),
                   plainto_tsquery('english', :keyword)) AS keyword_score
    FROM itembank
    WHERE to_tsvector('english', stem || ' ' || combined) @@ 
          plainto_tsquery('english', :keyword)
),
-- 2. Semantic similarity
semantic_results AS (
    SELECT 
        item_id,
        1 - (embedding <=> :query_embedding::vector) AS semantic_score
    FROM itembank
    WHERE embedding IS NOT NULL
)
SELECT 
    i.item_id,
    i.domain,
    i.topic,
    i.stem,
    i.rasch_b,
    i.pvalue,
    i.point_biserial,
    -- Weighted hybrid score
    COALESCE(k.keyword_score, 0) * 0.25 +  -- 25% keyword weight
    COALESCE(s.semantic_score, 0) * 0.75   -- 75% semantic weight
    AS final_score
FROM itembank i
LEFT JOIN keyword_results k ON i.item_id = k.item_id
LEFT JOIN semantic_results s ON i.item_id = s.item_id
WHERE k.item_id IS NOT NULL OR s.item_id IS NOT NULL
ORDER BY final_score DESC
LIMIT :top_k;
```

---

## Summary: When to Use Each Method

| Search Type | Use When | Pros | Cons |
|-------------|----------|------|------|
| **Pure Vector** | General semantic search | Fast, captures meaning | Misses exact keywords |
| **Pure Keyword** | Need exact term matching | Precise for specific terms | Misses synonyms/concepts |
| **Hybrid (FTS + Vector)** | Best overall results | Balances precision & recall | Slightly slower |
| **RRF** | Combining multiple methods | Robust, no weight tuning | More complex query |
| **Filtered Vector** | Domain-specific search | Targeted results | Smaller result set |
| **MMR** | Diverse results needed | Avoids redundancy | Slower, complex |

**For your NCLEX use case:** Use **Hybrid Search (Option A or C)** with 70-80% weight on semantic similarity and 20-30% on keyword matching for best results!
