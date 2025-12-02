"""
Generate embeddings for all items in the database that don't have them yet
"""
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import time

DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # Process items in batches

print("="*80)
print("EMBEDDING GENERATION FOR ITEMBANK")
print("="*80)

# Load the model
print("\nLoading SentenceTransformer model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Model loaded: {EMBEDDING_MODEL}")

# Connect to database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Check current status
cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NULL")
items_without_embeddings = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NOT NULL")
items_with_embeddings = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM itembank")
total_items = cur.fetchone()[0]

print(f"\n📊 Current Status:")
print(f"   Total items: {total_items}")
print(f"   With embeddings: {items_with_embeddings}")
print(f"   Without embeddings: {items_without_embeddings}")

if items_without_embeddings == 0:
    print("\n✅ All items already have embeddings!")
    cur.close()
    conn.close()
    exit()

# Confirm before proceeding
print(f"\n⚠️  This will generate embeddings for {items_without_embeddings} items.")
print(f"   Estimated time: ~{items_without_embeddings / 100:.1f} minutes")
response = input("\nProceed? (yes/no): ").strip().lower()

if response != 'yes':
    print("❌ Operation cancelled")
    cur.close()
    conn.close()
    exit()

# Fetch all items without embeddings
print("\n📥 Fetching items without embeddings...")
cur.execute("""
    SELECT ctid, item_id, combined
    FROM itembank
    WHERE embedding IS NULL
    ORDER BY item_id
""")
items = cur.fetchall()
print(f"✅ Fetched {len(items)} items")

# Process in batches
print(f"\n🔄 Generating embeddings (batch size: {BATCH_SIZE})...")
total_processed = 0
errors = []

for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i+BATCH_SIZE]
    
    try:
        # Extract text from batch
        ctids = [item[0] for item in batch]
        item_ids = [item[1] for item in batch]
        texts = [item[2] if item[2] else "" for item in batch]
        
        # Generate embeddings for batch
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        
        # Update database for each item in batch
        for ctid, item_id, embedding in zip(ctids, item_ids, embeddings):
            # Convert numpy array to list
            embedding_list = embedding.tolist()
            
            # Format as pgvector string
            vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding_list) + "]"
            
            # Update the specific row using ctid
            cur.execute("""
                UPDATE itembank 
                SET embedding = %s::vector
                WHERE ctid = %s
            """, (vec_str, ctid))
        
        # Commit after each batch
        conn.commit()
        total_processed += len(batch)
        
    except Exception as e:
        errors.append((i, str(e)))
        print(f"\n❌ Error in batch starting at index {i}: {e}")
        conn.rollback()
        continue

print(f"\n✅ Processing complete!")
print(f"   Successfully processed: {total_processed} items")
print(f"   Errors: {len(errors)}")

if errors:
    print("\n⚠️  Errors encountered:")
    for idx, error in errors[:10]:  # Show first 10 errors
        print(f"   Batch {idx}: {error}")

# Verify results
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NULL")
remaining = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NOT NULL")
completed = cur.fetchone()[0]

print(f"Items with embeddings: {completed}")
print(f"Items without embeddings: {remaining}")

if remaining == 0:
    print("\n🎉 SUCCESS! All items now have embeddings!")
else:
    print(f"\n⚠️  {remaining} items still need embeddings (likely due to errors)")

# Check for duplicates in the newly embedded items
print("\n" + "="*80)
print("CHECKING FOR DUPLICATES")
print("="*80)

cur.execute("""
    SELECT COUNT(*) 
    FROM (
        SELECT item_id 
        FROM itembank 
        GROUP BY item_id 
        HAVING COUNT(*) > 1
    ) AS duplicates
""")
duplicate_count = cur.fetchone()[0]

if duplicate_count > 0:
    print(f"⚠️  Found {duplicate_count} item_ids with duplicates")
    print("   Run cleanup_duplicates.py to fix this")
else:
    print("✅ No duplicates found")

cur.close()
conn.close()

print("\n" + "="*80)
print("EMBEDDING GENERATION COMPLETE")
print("="*80)
