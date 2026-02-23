# Implementation Plan: Pure AI LOFT Web Application

## 1. Goal Description
The user wants a dedicated, standalone **AI LOFT (Linear-on-the-fly) Web Application** built with Streamlit, moving completely away from imitating traditional deterministic Simultaneous CAT/ATA programs like the old `CBC_ATA.py`.

The new system must emphasize the AI-driven workflow:
1.  **AI Constraint Extraction:** The LLM reads natural language and specifies exactly how forms should be constructed (Domain counts, test length).
2.  **LOFT Master Bank Sampling:** The tracker filters available items and randomly samples them into an "Active Pool" using a user-defined **Multiplier** (e.g., 3x, 5x the test length).
3.  **MIP Assembly with Advanced Constraints:** The active pool is then solved deterministically for strict exactness. The MIP solver must now be upgraded to handle:
    *   **Testlets**: Items grouped by a single passage/stimulus. (e.g., `testlet_id` column). If a testlet is activated, its items must satisfy specific group constraints.
    *   **Enemy Items**: Mutually exclusive items (`enemy_ids` column).
    *   **Mean Difficulty**: Hard bounds or flexible ranges on the form's overall Rasch B mean.
    *   **3-Point TIF Evaluation**: TIF targets at the cut score (`theta`), and anchors at `theta - 1` and `theta + 1`.
4.  **Validation & Overlap:** Each form is validated sequentially. A **Form Similarity** check ensures identical form patterns are not repeatedly generated, complementing the global Item Usage Tracker.

---

## 2. Proposed Changes

### `AI_LOFT_App.py` (New File)
A completely fresh Streamlit dashboard customized exclusively for the AI LOFT workflow.
*   **[NEW] AI_LOFT_App.py**
    *   **Sidebar**: AI Provider Selection, API Key, LLM Prompt input box.
    *   **Advanced Engine Settings**: Multiplier Input (slider 2x-10x, default 4x), Max Form Overlap Threshold (%).
    *   **Execution**: Parses the LLM prompt, then runs the updated `sequential_loft_assembly` loop.
    *   **Visuals**: Clean output showing form generation iterations, TIF curves at the 3 evaluation points, overlap matrices, and exposure exhaustion.

### `ai_ata_engine.py` (Core Logic Upgrade)
*   **[MODIFY] ai_ata_engine.py**
    *   **`generate_active_pool`**: Expose the `multiplier` argument.
    *   **`assemble_single_form_mip`**: Add the following complex constraints mathematically via PuLP:
        1.  **3-Point TIF**: Target TIF minimums at `theta`, `theta-1`, `theta+1`.
        2.  **Enemy Constraint**: `x_i + x_j <= 1` for all listed enemies.
        3.  **Mean Difficulty**: Calculate the sum of `Rasch B * x_i`. Ensure it falls within the target range: `sum(b_i * x_i) / test_length >= min_mean` and `<= max_mean`.
        4.  **Testlets**: Identify unique `testlet_id` values. Create auxiliary binary vars for `y_t` (testlet active).
    *   **`sequential_loft_assembly`**: Implement the **Form Similarity (Overlap)** rejection logic. After a form is assembled, calculate Jaccard index with previous forms. If it fails, reject the active pool and resample.

---

## 3. Verification Plan
*   **Unit Verification**: Provide a mock item bank that includes `testlet_id` and `enemy_ids` to ensure the extended PuLP constraints trigger properly.
*   **UI Verification**: Run `streamlit run AI_LOFT_App.py` and execute an end-to-end prompt to ensure the 3x Multiplier and new parameters are correctly passed from the UI down to the core math engine.
