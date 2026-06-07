import numpy as np
import pandas as pd

def main():
    print("Generating 200-item mixed Rasch test parameters CSV...")
    np.random.seed(42)
    
    rows = []
    
    # 1. 100 Dichotomous Items
    # difficulties spaced normally across the trait range [-2.5, 2.5]
    dich_diffs = np.random.normal(0.0, 1.0, 100)
    for i in range(100):
        item_id = f"Item_{i+1:03d}"
        rows.append({
            "Item_ID": item_id,
            "Item_Type": "Dichotomous",
            "Difficulty": round(dich_diffs[i], 4),
            "Step_Difficulties": "",
            "Label": f"Dichotomous Item {i+1} (b = {dich_diffs[i]:.2f})"
        })
        
    # 2. 50 Rating Scale Model (RSM) Items (Polytomous with shared relative step thresholds)
    # Let's assume a 4-category scale (0, 1, 2, 3), meaning 3 relative step thresholds: [-1.2, 0.0, 1.2]
    # Each item has a unique location/difficulty.
    rsm_diffs = np.random.normal(0.2, 0.8, 50)
    shared_rsm_steps = [-1.2, 0.0, 1.2]
    rsm_steps_str = ";".join(map(str, shared_rsm_steps))
    
    for i in range(50):
        item_num = i + 101
        item_id = f"Item_{item_num:03d}"
        rows.append({
            "Item_ID": item_id,
            "Item_Type": "Polytomous",
            "Difficulty": round(rsm_diffs[i], 4),
            "Step_Difficulties": rsm_steps_str,
            "Label": f"RSM Item {item_num} (Location = {rsm_diffs[i]:.2f}, 4 categories)"
        })
        
    # 3. 50 Partial Credit Model (PCM) Items (Polytomous with unique absolute step difficulties)
    # We will generate varying number of categories (2 to 4 steps) with item-specific step difficulties
    for i in range(50):
        item_num = i + 151
        item_id = f"Item_{item_num:03d}"
        
        # Randomly choose number of steps between 2 and 4 (3 to 5 categories)
        n_steps = np.random.choice([2, 3, 4])
        
        # Generate item-specific absolute step difficulties
        # centered around a random item location
        item_center = np.random.uniform(-1.0, 1.2)
        step_deviations = np.sort(np.random.uniform(-1.5, 1.5, n_steps))
        abs_steps = np.round(item_center + step_deviations, 4)
        
        pcm_steps_str = ";".join(map(str, abs_steps))
        
        rows.append({
            "Item_ID": item_id,
            "Item_Type": "Polytomous",
            "Difficulty": "", # No overall difficulty column needed (PCM absolute steps)
            "Step_Difficulties": pcm_steps_str,
            "Label": f"PCM Item {item_num} (Absolute PCM steps, {len(abs_steps)+1} categories)"
        })
        
    df = pd.DataFrame(rows)
    output_filename = "rasch_mixed_200_items.csv"
    
    # Save with quoting for Label fields in case they contain commas (standard CSV formatting)
    df.to_csv(output_filename, index=False)
    print(f"Generated successfully and saved to {output_filename}")
    
    # Calculate some summary stats to print
    total_max_score = 100 * 1 + 50 * 3 + sum(len(row["Step_Difficulties"].split(";")) for row in rows[150:])
    print(f"Maximum test score (M): {total_max_score}")

if __name__ == "__main__":
    main()
