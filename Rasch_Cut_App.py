import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasch_mixed_item_cut as rmic
import io

# Set page config for a wide, professional dashboard layout
st.set_page_config(
    page_title="Rasch Mixed Item Cut score Converter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Default item bank to load if no file is uploaded (matching the notebook example)
DEFAULT_ITEMS = [
    {"difficulty": -1.2, "label": "Dichotomous Item 1"},
    {"difficulty": -0.5, "label": "Dichotomous Item 2"},
    {"difficulty": 0.0, "label": "Dichotomous Item 3"},
    {"difficulty": 0.5, "label": "Dichotomous Item 4"},
    {"difficulty": 1.0, "label": "Dichotomous Item 5"},
    {"difficulty": 1.8, "label": "Dichotomous Item 6"},
    {"steps": [-1.0, 0.5], "label": "PCM Item 7 (3 categories: 0, 1, 2)"},
    {"steps": [-0.5, 0.2, 1.2], "label": "PCM Item 8 (4 categories: 0, 1, 2, 3)"},
    {"difficulty": 0.2, "steps": [-0.8, 0.8], "label": "PCM Item 9 (RSM relative thresholds)"},
    {"steps": [0.0, 1.0, 2.0], "label": "PCM Item 10 (4 categories: 0, 1, 2, 3)"}
]

# Generate template CSV contents
# Min_Score: the lowest possible score a respondent can earn on an item.
# When Min_Score > 0 (non-zero base), use the 'Keep Base Score' option to match
# Winsteps behaviour (no RESCORE command). Use 'Recode to Zero' to shift all
# item scores so the minimum is 0 (equivalent to Winsteps RESCORE=0).
template_csv = """Item_ID,Item_Type,Difficulty,Step_Difficulties,Min_Score,Label
Item_1,Dichotomous,-1.2,,0,"Dichotomous Item 1"
Item_2,Dichotomous,-0.5,,0,"Dichotomous Item 2"
Item_3,Dichotomous,0.0,,0,"Dichotomous Item 3"
Item_4,Dichotomous,0.5,,0,"Dichotomous Item 4"
Item_5,Dichotomous,1.0,,0,"Dichotomous Item 5"
Item_6,Dichotomous,1.8,,0,"Dichotomous Item 6"
Item_7,Polytomous,,-1.0;0.5,0,"PCM Item 7 (3 categories: 0, 1, 2)"
Item_8,Polytomous,,-0.5;0.2;1.2,0,"PCM Item 8 (4 categories: 0, 1, 2, 3)"
Item_9,Polytomous,0.2,-0.8;0.8,0,"PCM Item 9 (RSM relative thresholds)"
Item_10,Polytomous,,0.0;1.0;2.0,0,"PCM Item 10 (4 categories: 0, 1, 2, 3)"
"""

import re

def _detect_base_from_label(label_str):
    """
    Auto-detect the minimum category score from an item label.
    Looks for patterns like '(2,3,4)' or '(0,1,2,3)' and returns the smallest
    category value found.  Returns 0 when no pattern is matched.
    """
    # Match (x,y,...,z) patterns — comma-separated integers inside parentheses
    m = re.search(r'\((\d+(?:\s*,\s*\d+)+)\)', label_str)
    if m:
        cats = [int(c.strip()) for c in m.group(1).split(",")]
        return min(cats)
    return 0

def parse_uploaded_csv(df):
    """
    Parses items from uploaded pandas DataFrame.
    Returns (items_list, min_scores) where min_scores is a list of per-item base scores.

    Base score detection priority:
      1. Explicit 'Min_Score' column in the CSV (highest priority).
      2. Auto-detection from 'Label' column patterns like '(2,3,4)'.
      3. Default to 0.
    """
    items_list = []
    min_scores = []
    
    # Required columns checks
    if "Item_ID" not in df.columns or "Item_Type" not in df.columns:
        raise ValueError("CSV file must contain at least 'Item_ID' and 'Item_Type' columns.")

    has_min_score_col = "Min_Score" in df.columns

    for _, row in df.iterrows():
        item_id = str(row["Item_ID"])
        item_type = str(row["Item_Type"]).strip().lower()
        
        difficulty = None
        if "Difficulty" in df.columns and pd.notna(row["Difficulty"]):
            difficulty = float(row["Difficulty"])
            
        steps = None
        if "Step_Difficulties" in df.columns and pd.notna(row["Step_Difficulties"]):
            steps = [float(x) for x in str(row["Step_Difficulties"]).split(";") if x.strip()]

        label = str(row["Label"]) if "Label" in df.columns and pd.notna(row["Label"]) else item_id

        # Min_Score detection: explicit column > label auto-detect > 0
        if has_min_score_col and pd.notna(row["Min_Score"]):
            min_score = int(float(row["Min_Score"]))
        else:
            min_score = _detect_base_from_label(label)
        
        item_dict = {"label": label}
        if item_type in ["dichotomous", "dich", "0/1"]:
            if difficulty is None:
                raise ValueError(f"Dichotomous item {item_id} must have a Difficulty value.")
            item_dict["difficulty"] = difficulty
        else: # Polytomous / PCM
            if steps is None:
                raise ValueError(f"Polytomous item {item_id} must have semicolon-separated Step_Difficulties.")
            item_dict["steps"] = steps
            if difficulty is not None:
                item_dict["difficulty"] = difficulty
                
        items_list.append(item_dict)
        min_scores.append(min_score)
        
    return items_list, min_scores

# Sidebar header & file uploader
st.sidebar.title("🎯 Control Panel")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Item Parameters CSV", type=["csv"])

# Download sample template button
st.sidebar.download_button(
    label="Download Template CSV",
    data=template_csv,
    file_name="rasch_items_template.csv",
    mime="text/csv"
)

# ── Base-score handling (Winsteps compatibility) ──────────────────────────────
st.sidebar.markdown("### Base Score Handling")
base_score_mode = st.sidebar.radio(
    "Minimum Category Score",
    options=["Keep Base Score (Winsteps default)", "Recode to Zero"],
    index=0,
    help=(
        "**Keep Base Score**: Raw scores include each item's minimum category value "
        "(e.g. if an item is scored 1–4, its contribution to the test total starts at 1). "
        "This matches Winsteps output when no RESCORE command is used.\n\n"
        "**Recode to Zero**: Shift every item so its minimum category = 0 before "
        "computing the TCC (equivalent to Winsteps RESCORE=0)."
    )
)
keep_base = (base_score_mode == "Keep Base Score (Winsteps default)")

# Parse parameters
items_to_use = DEFAULT_ITEMS
min_scores_list = [0] * len(DEFAULT_ITEMS)   # default: all items start at 0
is_uploaded = False

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        items_to_use, min_scores_list = parse_uploaded_csv(df_uploaded)
        is_uploaded = True
        st.sidebar.success("Item parameters file uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {str(e)}")
        st.sidebar.warning("Using default symmetric items instead.")
        min_scores_list = [0] * len(DEFAULT_ITEMS)

# Parse the items (step difficulties are always 0-based in the PCM engine)
parsed_items = rmic.parse_items(items_to_use)

# Per-item maximum score (number of steps above the base category)
item_max_steps = [len(steps) for steps in parsed_items]   # PCM engine max per item

# Total max score depends on base-score mode:
#   Keep Base Score → max = sum(min_score_i + steps_i)
#   Recode to Zero  → max = sum(steps_i)  [current behaviour]
total_base_offset = sum(min_scores_list)   # 0 when all min_scores are 0
if keep_base:
    max_score = sum(ms + ns for ms, ns in zip(min_scores_list, item_max_steps))
else:
    max_score = sum(item_max_steps)  # recode: ignore base offsets

# Add inputs in sidebar based on loaded test structure
st.sidebar.markdown("### Test Parameters")
raw_cut = st.sidebar.number_input(
    "Raw Cut Score", 
    min_value=0, 
    max_value=int(max_score), 
    value=int(max_score) // 2, 
    step=1
)

extrscore = st.sidebar.slider(
    "Extreme Score Adjustment",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
    help="Adjustment applied to 0 and maximum possible raw scores to enable finite logit estimates."
)

# ── Helper: convert a raw cut score to the 0-based score the PCM engine uses ─
# When keeping base scores, subtract the total minimum offset before estimation.
def effective_cut(raw, keep_base, total_base_offset, item_max_steps):
    """Return the 0-based score for Newton-Raphson estimation."""
    if keep_base:
        return max(0.0, float(raw) - total_base_offset)
    return float(raw)

# Effective 0-based max for the PCM engine (always sum of steps)
pcm_max_score = sum(item_max_steps)

st.sidebar.markdown("---")
st.sidebar.markdown("**About the Model:** Uses Andrich adjacent-category log-odds to construct the Master Partial Credit Model (PCM) Test Characteristic Curve, solving using the Newton-Raphson iteration solver.")

# Main dashboard layout
st.title("🎯 Rasch Mixed Item Cut Score Converter")
st.markdown("Easily calculate logit cut measures and test statistics under the Rasch Partial Credit Model.")

# 1. Summary Cards (KPI Metrics)
col1, col2, col3, col4 = st.columns(4)

total_items = len(items_to_use)
dich_count = sum(1 for item in items_to_use if "steps" not in item)
poly_count = total_items - dich_count

with col1:
    st.markdown("<div class='metric-card'><h4>Total Items</h4><h2>{}</h2></div>".format(total_items), unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><h4>Dichotomous Items</h4><h2>{}</h2></div>".format(dich_count), unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><h4>Polytomous Items</h4><h2>{}</h2></div>".format(poly_count), unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'><h4>Max Raw Score</h4><h2>{}</h2></div>".format(max_score), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Show base-score info banner when non-zero offsets are present ─────────────
if keep_base and total_base_offset > 0:
    st.info(
        f"📌 **Keep Base Score mode**: Total minimum score offset = **{total_base_offset}** "
        f"(sum of all item Min_Score values). "
        f"The PCM engine estimates on the 0-based scale "
        f"(raw − {total_base_offset}), then reports the Winsteps-equivalent raw score."
    )
elif not keep_base and total_base_offset > 0:
    st.warning(
        f"⚠️ **Recode to Zero mode**: Item base scores are ignored. "
        f"Max score = {max_score} (Winsteps max would be {max_score + total_base_offset})."
    )

# 2. Solver Result Panel
# Convert the Winsteps raw cut to the 0-based score the PCM engine expects
eff_raw_cut = effective_cut(raw_cut, keep_base, total_base_offset, item_max_steps)
res = rmic.raw_to_logit(eff_raw_cut, parsed_items, extrscore=extrscore)
logit_cut = res["logit"]
se_cut = res["se"]

# Compute test information at the cut score
_, final_slope = rmic.compute_tcc_and_slope(logit_cut, parsed_items)
tif_cut = final_slope

st.subheader("🎯 Mapped Logit Cut & Standard Error")
res_col1, res_col2, res_col3 = st.columns(3)

with res_col1:
    st.metric(
        label="Logit Cut Score (θ)",
        value="{:,.4f} logits".format(logit_cut),
        help="The ability measure corresponding to the target raw cut score."
    )
with res_col2:
    st.metric(
        label="Model Standard Error (SE)",
        value="{:,.4f} logits".format(se_cut),
        help="The precision of measurement at the cut score."
    )
with res_col3:
    st.metric(
        label="Test Information (I(θ))",
        value="{:,.4f}".format(tif_cut),
        help="Test information value at the cut score (higher information means higher measurement precision)."
    )

if res["converged"]:
    eff_adj = res['adjusted_score'] + (total_base_offset if keep_base else 0)
    st.info(
        f"Newton-Raphson solver converged successfully in {res['iterations']} iterations. "
        f"Effective 0-based adjusted score: {res['adjusted_score']:.2f}"
        + (f" (Winsteps-equivalent: {eff_adj:.2f})" if keep_base and total_base_offset > 0 else "")
    )
else:
    st.warning("Newton-Raphson solver failed to converge. Review inputs.")

# 3. Visualizations
st.subheader("📈 Visualization of TCC & TIF Curves")
# Pass the effective 0-based cut score to the plotting functions
fig = rmic.plot_tcc_and_tif(items_to_use, cut_score=eff_raw_cut)
st.pyplot(fig)

# 4. Item Bank Table (data.table style)
st.subheader("📋 Item Parameters Bank")
st.markdown("Displaying the parsed step difficulties for each item in the test form.")

item_table_rows = []
for idx, steps in enumerate(parsed_items):
    label = items_to_use[idx].get("label", f"Item {idx+1}")
    item_type = "Dichotomous" if len(steps) == 1 else "Polytomous (PCM)"
    ms = min_scores_list[idx] if idx < len(min_scores_list) else 0
    eff_cats = len(steps) + 1            # number of score categories (0-based)
    base_max = ms + len(steps)           # max score in original (Winsteps) scale
    item_table_rows.append({
        "Item Number": idx + 1,
        "Item Label": label,
        "Item Type": item_type,
        "Min Score": ms,
        "Max Score (Original)": base_max,
        "Categories": eff_cats,
        "Absolute Step Difficulties (δ_ij)": ", ".join(["{:.3f}".format(s) for s in steps])
    })
df_item_table = pd.DataFrame(item_table_rows)

# Interactive data.table
st.dataframe(df_item_table, use_container_width=True)

# 5. Conversion Table
st.subheader("📊 Raw-to-Logit Scoring Table")
if keep_base and total_base_offset > 0:
    st.markdown(
        "The complete scoring table for all possible raw scores on this test form. "
        f"**Raw Score (X)** shows the Winsteps-equivalent score "
        f"(includes base offset of {total_base_offset}). "
        "The PCM engine estimates on the 0-based scale internally."
    )
else:
    st.markdown("The complete scoring table for all possible raw scores on this test form.")

# Generate the 0-based conversion table (PCM engine)
conversion_df_raw = rmic.generate_conversion_table(items_to_use, extrscore=extrscore)

if keep_base and total_base_offset > 0:
    # Add the base offset back to the displayed raw scores so they match Winsteps
    conversion_df = conversion_df_raw.copy()
    conversion_df["Raw Score (X)"] = conversion_df_raw["Raw Score"] + total_base_offset
    conversion_df["Adjusted Score (X_adj)"] = conversion_df_raw["Adjusted Score"] + total_base_offset
    conversion_df["Ability Measure (θ)"] = conversion_df_raw["Logit Measure"]
    conversion_df["Model SE"] = conversion_df_raw["Model SE"]
    conversion_df["Converged"] = conversion_df_raw["Converged"]
    conversion_df["Iterations"] = conversion_df_raw["Iterations"]
    conversion_df = conversion_df[["Raw Score (X)", "Adjusted Score (X_adj)",
                                   "Ability Measure (θ)", "Model SE",
                                   "Converged", "Iterations"]]
else:
    conversion_df = conversion_df_raw.rename(columns={
        "Raw Score": "Raw Score (X)",
        "Adjusted Score": "Adjusted Score (X_adj)",
        "Logit Measure": "Ability Measure (θ)"
    })

st.dataframe(conversion_df, use_container_width=True)

# Download button for conversion table
csv_buffer = io.StringIO()
conversion_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Complete Conversion Table CSV",
    data=csv_buffer.getvalue(),
    file_name="rasch_conversion_table.csv",
    mime="text/csv"
)
