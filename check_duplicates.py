"""
Check for duplicate item_ids in the database
"""
import psycopg2

DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Check for duplicate item_ids
print("Checking for duplicate item_ids...")
cur.execute("""
    SELECT item_id, COUNT(*) as count
    FROM itembank
    WHERE embedding IS NOT NULL
    GROUP BY item_id
    HAVING COUNT(*) > 1
    ORDER BY count DESC
    LIMIT 10
""")

duplicates = cur.fetchall()
if duplicates:
    print(f"\n❌ Found {len(duplicates)} item_ids with duplicates:")
    for item_id, count in duplicates:
        print(f"  {item_id}: {count} rows")
    
    # Show details of first duplicate
    first_dup = duplicates[0][0]
    print(f"\n📋 Details of {first_dup}:")
    cur.execute("""
        SELECT item_id, domain, topic, stem, key
        FROM itembank
        WHERE item_id = %s
    """, (first_dup,))
    rows = cur.fetchall()
    for i, row in enumerate(rows, 1):
        print(f"\n  Row {i}:")
        print(f"    Domain: {row[1]}")
        print(f"    Topic: {row[2]}")
        print(f"    Stem: {row[3][:100]}...")
        print(f"    Key: {row[4]}")
else:
    print("✅ No duplicates found!")

# Check total count
cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NOT NULL")
total = cur.fetchone()[0]
print(f"\n📊 Total rows with embeddings: {total}")

cur.execute("SELECT COUNT(DISTINCT item_id) FROM itembank WHERE embedding IS NOT NULL")
unique = cur.fetchone()[0]
print(f"📊 Unique item_ids: {unique}")
print(f"📊 Duplicate rows: {total - unique}")

cur.close()
conn.close()
