import pandas as pd
import numpy as np
import re

def clean_enemy(x):
    if pd.isna(x) or x == "":
        return ""
    # Convert to string
    s = str(x)
    # Remove braces if present
    s = s.replace('{', '').replace('}', '')
    # Remove quotes if present
    s = s.replace("'", "").replace('"', "")
    return s.strip()

try:
    print("Loading item_bank_clean.csv...")
    df = pd.read_csv('item_bank_clean.csv')
    print(f"Loaded {len(df)} items.")
    print("Columns found:", list(df.columns))

    # Create output dataframe
    ata_df = pd.DataFrame()
    ata_df['ItemID'] = df['item_id']
    ata_df['Domain'] = df['domain']
    
    # Handle statistics
    ata_df['Pvalue'] = df['pvalue'].fillna(0.65) 
    ata_df['PBS'] = df['point_biserial'].fillna(0.35)
    ata_df['RaschB'] = df['rasch_b'].fillna(0.0)
    
    # Map and clean Enemy column
    if 'enemy' in df.columns:
        print("Processing 'enemy' column...")
        ata_df['Enemy'] = df['enemy'].apply(clean_enemy)
    else:
        print("Warning: 'enemy' column not found. Creating empty column.")
        ata_df['Enemy'] = ""

    # Replace NaN with empty string in Enemy column to prevent type errors
    ata_df['Enemy'] = ata_df['Enemy'].fillna("")
    
    # Ensure all numeric columns don't have NaN
    ata_df['Pvalue'] = ata_df['Pvalue'].fillna(0.65)
    ata_df['PBS'] = ata_df['PBS'].fillna(0.35)
    ata_df['RaschB'] = ata_df['RaschB'].fillna(0.0)
    
    # Save to CSV with na_rep to handle any remaining NaN
    output_file = 'ATA_sample.csv'
    ata_df.to_csv(output_file, index=False, na_rep="")
    print(f"Successfully created {output_file}")
    print("Columns:", list(ata_df.columns))
    print("Sample rows:")
    print(ata_df.head())
    
    # Show some non-empty enemies to verify
    print("\nSample non-empty enemies:")
    print(ata_df[ata_df['Enemy'] != ""]['Enemy'].head())

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
