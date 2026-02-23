import pandas as pd
from ai_ata_engine import sequential_loft_assembly

def test_engine():
    print("Loading test item bank...")
    try:
        item_bank = pd.read_csv('item_bank_hosted2.csv')
    except Exception as e:
        print(f"Failed to load item bank: {e}")
        return

    item_bank['domain'] = item_bank['domain'].fillna('Unspecified')
    
    available_domains = item_bank['domain'].unique()
    print(f"Available Domains in pool: {available_domains[:3]}...")
    
    print("\n--- Running AI ATA Sequential Engine (Generator) ---")
    gen = sequential_loft_assembly(
        item_bank=item_bank,
        user_prompt="Build me 3 math forms of 10 items each, targeting theta 0.5" 
    )
    
    forms = []
    usage = {}
    for step in gen:
        if step['step'] == 'sampling':
            print(f"  Sampling pool for Form {step['form_idx']}...")
        elif step['step'] == 'form_complete':
            f = step['latest_form']
            print(f"  ✅ Form {step['form_idx']} | Length: {f['metrics']['length']} | Primary TIF: {f['metrics']['primary_tif']:.2f} | Mean B: {f['metrics']['mean_b']:.2f}")
            forms = step['forms']
            usage = step['usage_stats']
        elif step['step'] == 'warning':
            print(f"  ⚠️ {step['message']}")
        elif step['step'] == 'error':
            print(f"  🛑 {step['message']}")
        elif step['step'] == 'finished':
            forms = step['forms']
            usage = step['usage_stats']
    
    print(f"\nSuccessfully built {len(forms)} out of 3 forms requested.")
    
    used_items = {k: v for k, v in usage.items() if v > 0}
    max_usage = max(used_items.values()) if used_items else 0
    print(f"\nExposure Control Metrics:")
    print(f"Total Unique Items Used: {len(used_items)}")
    print(f"Max times any single item was used (Global limit is 2): {max_usage}")

if __name__ == '__main__':
    test_engine()
