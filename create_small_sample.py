import pandas as pd

# Read the full ATA_sample.csv
df = pd.read_csv('ATA_sample.csv')

print(f"Original dataset: {len(df)} items")
print(f"Domains: {df['Domain'].value_counts().to_dict()}")

# Take a stratified sample by domain (50 items per domain, max 250 total)
sample_df = df.groupby('Domain', group_keys=False).apply(
    lambda x: x.sample(min(50, len(x)), random_state=42)
).reset_index(drop=True)

# Limit to 250 items total
sample_df = sample_df.head(250)

# Save to new file
sample_df.to_csv('ATA_sample_small.csv', index=False)

print(f"\nCreated ATA_sample_small.csv with {len(sample_df)} items")
print(f"Domains in sample: {sample_df['Domain'].value_counts().to_dict()}")
print("\nYou can now upload ATA_sample_small.csv in the Streamlit app!")
