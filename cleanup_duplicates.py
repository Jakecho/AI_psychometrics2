"""
Analyze and clean duplicate item_ids in the database
"""
import psycopg2
import hashlib

DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

def get_content_hash(row):
    """Create hash of item content to detect if duplicates are truly identical"""
    # Combine all content fields (excluding item_id and embedding)
    content = f"{row[1]}|{row[2]}|{row[3]}|{row[4]}|{row[5]}|{row[6]}|{row[7]}|{row[8]}|{row[9]}"
    return hashlib.md5(content.encode()).hexdigest()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("="*80)
print("ANALYZING DUPLICATE ITEMS IN DATABASE")
print("="*80)

# Get all duplicate item_ids
cur.execute("""
    SELECT item_id, COUNT(*) as count
    FROM itembank
    WHERE embedding IS NOT NULL
    GROUP BY item_id
    HAVING COUNT(*) > 1
    ORDER BY item_id
""")

duplicates = cur.fetchall()
print(f"\n📊 Found {len(duplicates)} item_ids with duplicates\n")

identical_duplicates = []
different_duplicates = []

# Analyze each duplicate
for item_id, count in duplicates:
    cur.execute("""
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", "choice_C", "choice_D", 
               key, rationale, rasch_b, pvalue, point_biserial, combined
        FROM itembank
        WHERE item_id = %s
        ORDER BY item_id
    """, (item_id,))
    
    rows = cur.fetchall()
    
    # Get content hashes
    hashes = [get_content_hash(row) for row in rows]
    
    if len(set(hashes)) == 1:
        # All duplicates are identical
        identical_duplicates.append((item_id, count, rows))
        print(f"✓ {item_id}: {count} IDENTICAL duplicates (will delete extras)")
    else:
        # Duplicates have different content
        different_duplicates.append((item_id, count, rows))
        print(f"⚠ {item_id}: {count} duplicates with DIFFERENT content (will reassign IDs)")
        for i, row in enumerate(rows, 1):
            print(f"  Version {i}: Domain={row[1]}, Topic={row[2]}")
            print(f"            Stem: {row[3][:80]}...")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Identical duplicates (to delete): {len(identical_duplicates)} items")
print(f"Different content (to reassign):  {len(different_duplicates)} items")

# Show action plan
if different_duplicates:
    print("\n" + "="*80)
    print("REASSIGNMENT PLAN FOR DIFFERENT CONTENT")
    print("="*80)
    
    # Get current max item number
    cur.execute("""
        SELECT item_id FROM itembank 
        WHERE item_id ~ '^NCX[0-9]+$'
        ORDER BY CAST(SUBSTRING(item_id FROM 4) AS INTEGER) DESC 
        LIMIT 1
    """)
    max_id = cur.fetchone()
    if max_id:
        current_max = int(max_id[0][3:])  # Extract number from NCX0XXX
    else:
        current_max = 0
    
    next_number = current_max + 1
    
    for item_id, count, rows in different_duplicates:
        print(f"\n{item_id} has {count} different versions:")
        print(f"  Keep original: {item_id} (first occurrence)")
        for i in range(1, count):
            new_id = f"NCX{next_number:04d}"
            print(f"  Reassign copy {i+1}: {item_id} → {new_id}")
            next_number += 1

# Confirm before proceeding
print("\n" + "="*80)
response = input("\nProceed with cleanup? (yes/no): ").strip().lower()

if response != 'yes':
    print("❌ Cleanup cancelled")
    cur.close()
    conn.close()
    exit()

print("\n" + "="*80)
print("EXECUTING CLEANUP")
print("="*80)

# Step 1: Delete identical duplicates (keep first occurrence)
deleted_count = 0
for item_id, count, rows in identical_duplicates:
    # Get the ctid (physical row identifier) of all rows
    cur.execute("""
        SELECT ctid FROM itembank WHERE item_id = %s
    """, (item_id,))
    ctids = cur.fetchall()
    
    # Delete all but the first
    for i in range(1, len(ctids)):
        cur.execute("DELETE FROM itembank WHERE ctid = %s", (ctids[i][0],))
        deleted_count += 1
    
    print(f"✓ Deleted {count-1} duplicate(s) of {item_id}")

conn.commit()
print(f"\n✅ Deleted {deleted_count} identical duplicate rows")

# Step 2: Reassign IDs for different content
cur.execute("""
    SELECT item_id FROM itembank 
    WHERE item_id ~ '^NCX[0-9]+$'
    ORDER BY CAST(SUBSTRING(item_id FROM 4) AS INTEGER) DESC 
    LIMIT 1
""")
max_id = cur.fetchone()
if max_id:
    next_number = int(max_id[0][3:]) + 1
else:
    next_number = 1

reassigned_count = 0
for item_id, count, rows in different_duplicates:
    # Get ctids for this item_id
    cur.execute("""
        SELECT ctid FROM itembank WHERE item_id = %s
    """, (item_id,))
    ctids = cur.fetchall()
    
    print(f"\n{item_id}:")
    print(f"  Keeping first occurrence as {item_id}")
    
    # Reassign all duplicates except the first
    for i in range(1, len(ctids)):
        new_id = f"NCX{next_number:04d}"
        cur.execute("""
            UPDATE itembank 
            SET item_id = %s 
            WHERE ctid = %s
        """, (new_id, ctids[i][0]))
        print(f"  ✓ Reassigned duplicate {i+1} to {new_id}")
        next_number += 1
        reassigned_count += 1

conn.commit()
print(f"\n✅ Reassigned {reassigned_count} items with different content")

# Verify cleanup
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

cur.execute("""
    SELECT COUNT(*) as total,
           COUNT(DISTINCT item_id) as unique_items
    FROM itembank
    WHERE embedding IS NOT NULL
""")
result = cur.fetchone()
print(f"Total rows: {result[0]}")
print(f"Unique item_ids: {result[1]}")

if result[0] == result[1]:
    print("\n✅ SUCCESS: All item_ids are now unique!")
else:
    print(f"\n⚠ WARNING: Still have {result[0] - result[1]} duplicate rows")

cur.close()
conn.close()

print("\n" + "="*80)
print("CLEANUP COMPLETE")
print("="*80)
