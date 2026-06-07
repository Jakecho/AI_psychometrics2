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
template_csv = """Item_ID,Item_Type,Difficulty,Step_Difficulties,Label
Item_1,Dichotomous,-1.2,,"Dichotomous Item 1"
Item_2,Dichotomous,-0.5,,"Dichotomous Item 2"
Item_3,Dichotomous,0.0,,"Dichotomous Item 3"
Item_4,Dichotomous,0.5,,"Dichotomous Item 4"
Item_5,Dichotomous,1.0,,"Dichotomous Item 5"
Item_6,Dichotomous,1.8,,"Dichotomous Item 6"
Item_7,Polytomous,,-1.0;0.5,"PCM Item 7 (3 categories: 0, 1, 2)"
Item_8,Polytomous,,-0.5;0.2;1.2,"PCM Item 8 (4 categories: 0, 1, 2, 3)"
Item_9,Polytomous,0.2,-0.8;0.8,"PCM Item 9 (RSM relative thresholds)"
Item_10,Polytomous,,0.0;1.0;2.0,"PCM Item 10 (4 categories: 0, 1, 2, 3)"
"""

def parse_uploaded_csv(df):
    """
    Parses items from uploaded pandas DataFrame.
    """
    items_list = []
    
    # Required columns checks
    if "Item_ID" not in df.columns or "Item_Type" not in df.columns:
        raise ValueError("CSV file must contain at least 'Item_ID' and 'Item_Type' columns.")
        
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
        
    return items_list

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

# Parse parameters
items_to_use = DEFAULT_ITEMS
is_uploaded = False

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        items_to_use = parse_uploaded_csv(df_uploaded)
        is_uploaded = True
        st.sidebar.success("Item parameters file uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {str(e)}")
        st.sidebar.warning("Using default symmetric items instead.")

# Parse the items
parsed_items = rmic.parse_items(items_to_use)
max_score = sum(len(steps) for steps in parsed_items)

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

# 2. Solver Result Panel
res = rmic.raw_to_logit(raw_cut, parsed_items, extrscore=extrscore)
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
    st.info(f"Newton-Raphson solver converged successfully in {res['iterations']} iterations. Target adjusted score: {res['adjusted_score']:.2f}")
else:
    st.warning("Newton-Raphson solver failed to converge. Review inputs.")

# 3. Visualizations
st.subheader("📈 Visualization of TCC & TIF Curves")
fig = rmic.plot_tcc_and_tif(items_to_use, cut_score=raw_cut)
st.pyplot(fig)

# 4. Item Bank Table (data.table style)
st.subheader("📋 Item Parameters Bank")
st.markdown("Displaying the parsed step difficulties for each item in the test form.")

item_table_rows = []
for idx, steps in enumerate(parsed_items):
    label = items_to_use[idx].get("label", f"Item {idx+1}")
    item_type = "Dichotomous" if len(steps) == 1 else "Polytomous (PCM)"
    item_table_rows.append({
        "Item Number": idx + 1,
        "Item Label": label,
        "Item Type": item_type,
        "Categories (Max)": len(steps) + 1,
        "Absolute Step Difficulties (δ_ij)": ", ".join(["{:.3f}".format(s) for s in steps])
    })
df_item_table = pd.DataFrame(item_table_rows)

# Interactive data.table
st.dataframe(df_item_table, use_container_width=True)

# 5. Conversion Table
st.subheader("📊 Raw-to-Logit Scoring Table")
st.markdown("The complete scoring table for all possible raw scores on this test form.")

conversion_df = rmic.generate_conversion_table(items_to_use, extrscore=extrscore)
conversion_df.columns = ["Raw Score (X)", "Adjusted Score (X_adj)", "Ability Measure (θ)", "Model SE", "Converged", "Iterations"]

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
