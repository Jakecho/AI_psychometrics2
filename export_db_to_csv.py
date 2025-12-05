import psycopg2
import pandas as pd
import json

# DB Config
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

def export_to_csv():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        query = """
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
               point_biserial, embedding
        FROM itembank
        WHERE embedding IS NOT NULL
        ORDER BY item_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Save to CSV
        output_file = "item_bank_hosted.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully exported {len(df)} items to {output_file}")
        
    except Exception as e:
        print(f"Error exporting database: {e}")

if __name__ == "__main__":
    export_to_csv()
