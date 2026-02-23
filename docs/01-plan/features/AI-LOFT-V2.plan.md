# Implementation Plan: AI LOFT Web Application V2 (BKIT PDCA)

## 1. Goal Description
The user requested a massive enhancement to the standalone AI LOFT Engine (`AI_LOFT_App.py` and `ai_ata_engine.py`). They want it to support an exhaustive list of psychometric constraints typical of high-stakes CAT/ATA programs, while strictly maintaining the AI-driven Linear-on-the-fly (LOFT) methodology.

The updated process flow must strictly follow:
**Sampling (Multiplier) ➡️ Form Assembly (MIP) ➡️ Validation ➡️ Usage Update ➡️ Dashboard**

All of these features must be visible in a real-time monitoring dashboard.

---

## 2. Proposed Changes

### `ai_ata_engine.py` (Core MIP Upgrades)
The CBC solver inside `assemble_single_form_mip` must be expanded to handle:
*   **3-Point Evaluation with Tolerances**: 
    *   Target Information Function (TIF) at `theta, theta-1, theta+1` + **Tolerance limits**.
    *   Target Characteristic Curve (TCC / Expected Score) at `theta, theta-1, theta+1` + **Tolerance limits**.
*   **Item Feature Constraints**:
    *   **Image**: Min/Max count of items requiring images (`has_image`).
    *   **Audio**: Min/Max count of items requiring audio (`has_audio`).
*   **Psychometric/Statistical Constraints**:
    *   **Rasch B Category Control**: (`RaschB_cat`) Min/max item bounds for specific difficulty buckets.
*   **Security/Format Constraints**:
    *   **Common Items**: Force-include specific `item_id`s in the generated form.
    *   **Enemies & Testlets**: (Already drafted, but ensure they seamlessly integrate with the new constraints).
    *   **Item Exposure Control**: (Already active, but needs tracking hooks for the dashboard).

### `AI_LOFT_App.py` (Real-Time UI Dashboard)
The Streamlit app will be overhauled to become a true monitoring dashboard.
*   **Sidebar Enhancements**: 
    *   Add explicit toggles and inputs for TCC targets, TIF/TCC tolerances, Common Items, Image/Audio constraints, and Rasch B Categories (mirroring the depth of `CBC_ATA.py`).
    *   Keep the **Multiplier** slider prominent for the Active Pool sampling phase.
*   **Real-Time Monitoring Dashboard**:
    *   Use Streamlit `st.empty()` or real-time metric blocks to show live updates during the assembly loop.
    *   **Metrics**: 
        1. Number of forms successfully assembled.
        2. Live Item Usage counts (Max exposure vs current).
        3. Bank Utilization % (How much of the master pool is exhausted/locked).

---

## 3. Verification Plan
*   **Math Verification**: Inject mock `has_image`, `has_audio`, and `RaschB_cat` columns into the `item_bank_hosted2.csv` DataFrame during testing if they do not exist, ensuring the PuLP math doesn't crash on standard datasets.
*   **Flow Verification**: Ensure the console logs and Streamlit UI strictly follow the required sequence: `Sampling -> Assembly -> Validation -> Update -> UI`.
*   **UI Verification**: Boot `AI_LOFT_App.py` on port 8502 and visually confirm the real-time metrics update as each form is sequentially built.
