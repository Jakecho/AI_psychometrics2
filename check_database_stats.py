"""
Check total number of items in the database
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

print("="*80)
print("DATABASE STATISTICS")
print("="*80)

# Total rows in itembank table
cur.execute("SELECT COUNT(*) FROM itembank")
total_rows = cur.fetchone()[0]
print(f"\nTotal rows in itembank table: {total_rows}")

# Total rows with embeddings
cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NOT NULL")
total_with_embeddings = cur.fetchone()[0]
print(f"Rows with embeddings: {total_with_embeddings}")

# Total rows without embeddings
cur.execute("SELECT COUNT(*) FROM itembank WHERE embedding IS NULL")
total_without_embeddings = cur.fetchone()[0]
print(f"Rows without embeddings: {total_without_embeddings}")

# Unique item_ids
cur.execute("SELECT COUNT(DISTINCT item_id) FROM itembank")
unique_items = cur.fetchone()[0]
print(f"\nUnique item_ids: {unique_items}")

# Check for any duplicates
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
print(f"Item_ids with duplicates: {duplicate_count}")

# Item ID range
cur.execute("""
    SELECT MIN(item_id), MAX(item_id) 
    FROM itembank 
    WHERE item_id ~ '^NCX[0-9]+$'
""")
min_id, max_id = cur.fetchone()
print(f"\nItem ID range: {min_id} to {max_id}")

# Domain breakdown
print("\n" + "="*80)
print("BREAKDOWN BY DOMAIN")
print("="*80)
cur.execute("""
    SELECT domain, COUNT(*) as count
    FROM itembank
    GROUP BY domain
    ORDER BY count DESC
""")
domains = cur.fetchall()
for domain, count in domains:
    print(f"{domain}: {count} items")

# Topic breakdown (top 10)
print("\n" + "="*80)
print("TOP 10 TOPICS")
print("="*80)
cur.execute("""
    SELECT topic, COUNT(*) as count
    FROM itembank
    GROUP BY topic
    ORDER BY count DESC
    LIMIT 10
""")
topics = cur.fetchall()
for topic, count in topics:
    print(f"{topic}: {count} items")

cur.close()
conn.close()

print("\n" + "="*80)
