# MASTER IMPLEMENTATION PROMPT v2 — Reviewed for Engineering, Psychometrics & SME Practice

**Review status:** Cross-role specification review completed 2026-11-22; this is a revised *implementation specification*, not an implemented application.  
**Intended executor:** Claude Code, OpenAI Codex, or a comparable autonomous coding agent.  
**Artifact type:** Executable implementation specification and acceptance contract, **not** a user-facing essay.  
**Target:** Streamlit Community Cloud, local VS Code, or other small Python hosting environment.  
**Scope:** Less than 1,000 user-uploaded assessment items; no shipped NCLEX simulation data; no relational/vector database; OpenAI API for embeddings and structured cognitive feature extraction.

## 0. Mandatory agent operating instructions

Act as an integrated team of **senior Python/Streamlit engineer, psychometrician, clinical/assessment subject-matter expert (SME), security/privacy reviewer, and QA engineer**. Deliver an actual functioning proof-of-concept application and test suite, not merely design advice. Separate your implementation work into coherent phases, implement and test each phase, then perform a final cross-role review. Respect the repository's existing conventions where useful; do not blindly copy outdated or insecure choices.

### Evidence and workflow rules

1. **Inspect the source code first.** If repository access is available, examine the files listed in Section 1, including UI code, search algorithms, difficulty and classification logic, data contracts, requirements, and existing tests. If unavailable, create a clean standalone project using the specification below, and document what could not be verified. Do not hallucinate source behavior.
2. **Implement, run, and verify.** After each phase, run unit tests and static/syntax checks. Fix defects and rerun. End with a feature-by-feature status report: implemented, partially implemented, blocked, or unimplemented, with evidence and commands.
3. **Do not overwrite the existing app or push to GitHub without authorization.** Prefer a **new app file and branch/repository**, retaining original files for reference. If the user explicitly authorizes modifications, create a branch and make scoped commits rather than silently replacing files.
4. **Never expose or reuse credentials found in old source files.** The historical projects contain hard-coded connection settings. Replace with environment variables, `st.secrets`, or secure user-provided session credentials. No secrets in commits, logs, screenshots, exports, or test fixtures. Recommend rotation for any already-exposed credential without repeating its value.
5. **Do not ship real confidential or proprietary item content.** All tests, screenshots, and demonstrations must use independently invented synthetic examples unless the user supplies data and affirmatively authorizes its use. Make transmission of uploaded item text to an external API transparent and gated by an explicit consent action.
6. **Preserve all original user requirements.** Retrieval must be cosine-first; keyword search is optional, never a replacement for cosine. All feature categories must be shown, with **no fixed limit on feature count**. The SME's 1–5 importance rating is independent of how many features happen to exist.
7. When a missing detail can reasonably be decided, choose the simplest safe POC default, note it, and proceed. Ask a question only for a genuinely blocking ambiguity, secret, access, or external authorization.
8. Prefer reproducible, easily deployable Python code over large infrastructure. Avoid overengineering; ensure deterministic operations where possible.

### Mandatory priority and review gates (v2)

Treat the following as **blocking requirements** (P0): (a) no cross-user content or credential leakage; (b) explicit external-API consent; (c) task confirmation before sibling-feature frequency; (d) provable item/option provenance with explicit-versus-inferred evidence; (e) no comparisons of item difficulty estimates from unlinked measurement scales; (f) no irreversible overwrite of SME decisions; and (g) unique item/feature counting.

**P1 complete POC:** reliable arbitrary-column CSV/XLSX import, embeddings and exact cosine retrieval, item comparison, structured four-space decomposition, editable SME review and frequency table, metadata-colorable PCA, exports, elementary difficulty/classification/enemy heuristics. **P2 optional:** BM25/RRF, UMAP, feature-definition embeddings, advanced clustering, more intricate enemy rules. Implement P0 then P1 before polishing P2. All user-requested categories remain in scope; mark any undelivered P1 feature as incomplete, not implemented.

At each phase apply a three-role review (developer, psychometrician, item-writing SME) with an issue register: issue ID, severity P0/P1/P2, affected file, reproduced evidence, correction, test, and reviewer signoff. Unresolved P0 issues block any claim of readiness. Do not invent approvals from real SMEs: simulation/mock reviews are explicitly labeled simulated.

## 1. Existing project references and important findings

Inspect these files from the **user-connected GitHub projects** when access is available:

- `https://github.com/Jakecho/nclex-item-generator/blob/main/AIG_NCLEX.py` — includes a system prompt and user-prompt builder for NCLEX item generation. Use this as a reference for typed output and SME review, **not** as a requirement to generate NCLEX items.
- `https://github.com/Jakecho/nclex-item-generator/blob/main/NCLEX_RN_VectorDB.py` — legacy PostgreSQL/pgvector Streamlit item search, semantic retrieval, weighted hybrid search, RRF, Top K/threshold filtering, item previews, and export.
- `https://github.com/Jakecho/AI_psychometrics2/blob/main/NCLEX_RN_VectorDB_v2.py` — includes **Semantic Similarity Search**, **Item Difficulty Prediction**, and **Item Classification Prediction** tabs. Difficulty is a simple average of neighbor Rasch difficulties; classification is a neighbor-domain majority vote. Treat both as heuristics, not validated predictive models.
- `https://github.com/Jakecho/AI_psychometrics2/blob/main/NCLEX_RN_VectorDB_v3.py` — CSV-based fallback using stored embeddings, same general search workflow, and an `enemy` metadata column. Existing enemy metadata does **not** establish automated enemy detection.
- `https://github.com/Jakecho/nclex-item-generator/blob/main/README.md` — original app description, data schema, and deployment examples.

**Important replacement decisions:** No assumed fixed `itembank` SQL table; no reliance on a particular NCLEX CSV; no localhost database; no password in source; no 384-dimensional legacy embedding assumption. Do not claim the supplied Streamlit URL has been conclusively matched to a repository unless deployment settings substantiate it.

## 2. Product objective and required functional areas

Build a **generic, database-free item bank analytics and cognitive feature decomposition POC** with these cohesive capabilities:

1. **Import and Prepare:** CSV/XLSX upload; detect and select columns; validate and normalize records; inspect and filter metadata; map item IDs, stem, options/key, domain/task, difficulty and other fields.
2. **Embedding Generation:** OpenAI Embeddings API for user-selected text columns; batch generation; cache/reuse/export; preserve model configuration and fingerprints.
3. **Cosine Similarity Retrieval:** Exact in-memory cosine similarity on up to 1,000 item vectors; free-text target or item-ID target; Top K, minimum cosine threshold, or both; unique candidate items; optional BM25 and optional hybrid fusion.
4. **Task Validation:** AI proposes the target clinical/assessment task and domain, compares candidates with target item; SME verifies which candidates genuinely measure the same task.
5. **Key-Feature Cognitive Decomposition:** AI extracts unique **declarative key features**, **procedural key features** (one dominant and/or multiple defensible candidate procedures), **instance/context features**, and **distractor features**. Each feature is traceable to source items/options, with literal text evidence distinguished from reasoned task-based inference; SME accepts, rejects, edits, and rates importance.
6. **Distractor Review and Space:** Display **all** options for each unique sibling item; mark key vs distractors; propose misconception/incorrect-reasoning categories, deduplicate canonical distractor features, and show option-level provenance.
7. **Feature Frequencies:** Count each canonical feature **once per distinct accepted sibling item**; show numerator, denominator, percentage, IDs, and SME rating.
8. **Embedding & Feature Visualization:** PCA, t-SNE and optional UMAP 2D item scatterplot, with user-selected metadata color encoding, hover, filter, linked item details and target/candidate highlighting; separate interpretable feature incidence/heatmap/graph views.
9. **Difficulty Prediction:** Similarity-based **heuristic** prediction from uploaded empirical difficulty labels (e.g. Rasch `b`), with explicit limitations, neighbor statistics, and holdout diagnostics when data permit.
10. **Item Classification:** Predict categorical domain/task/blueprint label from labeled neighbors, show evidence and diagnostic confidence, allow abstention and SME override.
11. **Enemy-Item Candidate Detection:** Flag duplicates, stem/option overlap, same-task near-duplicates, answer leakage, or unsafe co-administration patterns for SME review; never automatically certify true enemies from similarity alone.
12. **Export/Import Results:** CSV/Excel for search results, unique items, reviewed features, evidence maps, annotations, and session/project bundle with vectors/model IDs/settings (excluding API secrets). Optionally provide JSON export for downstream use.

The UI should be organized around one main sequential workflow: **Upload → Map Columns → Embed → Search → Confirm Same Task → Decompose → SME Review → Visualize / Export**. Ancillary analytical tabs may branch after search.

## 3. Deployment and technical constraints

- Python 3.11+ (choose a tested Streamlit Cloud-supported version).
- Streamlit + pandas + NumPy + OpenAI official Python SDK + scikit-learn + Plotly + openpyxl. Optional `rank-bm25` for lexical search and `umap-learn` for UMAP; gracefully disable optional features when packages are missing.
- **No** SQL, PostgreSQL, pgvector, Pinecone, Chroma, Qdrant, FAISS, long-running server, local text-generation model, or separate vector DB requirement. For N <= 1,000, exact vector matrix multiplication is adequate.
- Embedding default `text-embedding-3-small` (default dimensionality **1536**, not the legacy MiniLM 384). Do not hard-code dimension: determine from API output, enforce consistency, and optionally support documented dimensionality reduction in API settings where the model permits it.
- Chat/reasoning model for decomposition: user-configurable OpenAI model ID supported by the account; use structured JSON-schema output where supported and robust parse/validate/retry behavior otherwise. Never assume a specific model is universally available.
- Explicit and understandable API usage consent, progress, cancellation where feasible, error reporting, and optional cost/usage accounting. Separate consent for embedding and LLM decomposition; show text-preview/sample and approximate token/request budget before sending. Limit LLM decomposition to the target plus SME-confirmed subset (never transmit 1,000 raw items in one request). Bounded chunking with cross-chunk canonicalization and provenance; preserve partial validated output after failures. If API fails, preserve existing vectors and reviews and allow retry without corrupting state.
- Use `st.session_state` for **per-user data**, mapping, indices, SME edits, target, accepted candidates, and reviewed features. `st.cache_data` and `st.cache_resource` are shared across users by default: NEVER place user-uploaded text, vectors, API keys, review annotations, model responses, or user-specific hashes/derived data in a global cache. Store sensitive and user-specific values solely in `st.session_state` or use demonstrably isolated per-session facilities. Global caches are permitted only for generic, public, immutable resources. Explicitly test user A/user B isolation. Session state is ephemeral and may reset when the WebSocket/session is lost; offer a download/reimport bundle.
- File data and generated vectors are ephemeral without download; clearly communicate that Streamlit session resets lose unsaved state.
- Avoid expensive global 2D visualization recalculations when unrelated controls change. Cache transforms by dataset/embedding fingerprint, method, parameters and random seed.

## 4. Project layout and architecture

Create an understandable modular structure (adapt if the repository already has equivalent modules):

```text
item-analytics-poc/
  app.py
  requirements.txt
  README.md
  .gitignore
  .streamlit/config.toml              # non-secret settings only
  .streamlit/secrets.toml.example    # placeholders only; do not commit real secrets
  src/
    schemas.py                      # Pydantic/dataclasses + validation
    ingest.py                       # CSV/Excel parsing and column mapping
    text_builder.py                 # deterministic selected-column concatenation
    embedding.py                    # OpenAI batching, validation, cache manifest
    retrieval.py                    # cosine, lexical option, ranking, filters
    task_alignment.py               # task inference and SME confirmation
    decomposition.py                # LLM prompts, structured parse, dedupe
    feature_frequency.py            # canonical incidence and counts
    distractors.py                  # keys/options and distractor mappings
    predictions.py                  # difficulty/classification heuristics
    enemies.py                      # candidate flags and provenance
    visualization.py                # PCA, t-SNE, UMAP, categorical/numeric color
    storage.py                      # session state / portable project bundle
    ui/                             # reusable Streamlit components (optional)
  prompts/
    target_task.md
    feature_decomposition.md
    canonicalization.md
  tests/
    conftest.py
    test_ingest.py
    test_embedding.py
    test_retrieval.py
    test_task_alignment.py
    test_decomposition.py
    test_frequency.py
    test_predictions.py
    test_visualization.py
    test_security.py
    test_app_smoke.py
  samples/
    synthetic_items.csv             # created solely for tests; no real items
```

Prefer pure functions for ingestion, cosine scores, feature incidence, and predicted statistics. Keep OpenAI calls isolated behind a mockable adapter. Avoid multipage app complexity unless it meaningfully improves user experience.

## 5. Data import, schema inference and column mapping

### Input modes
- Upload `.csv` (BOM/UTF-8 handling and clear decode errors), `.xlsx` (select sheet if more than one); optional `.xls` when dependency installed.
- Support first-row header with detection; show row count, columns, types, null percentages, example values, nonempty counts, and duplicate item IDs.
- User can choose **one or multiple** text columns for embeddings (e.g. vignette, stem, lead-in, options); can choose columns to display even if not embedded. Show exact concatenation order and preview.
- User maps `item_id`, `stem`, key, answer-option columns (A–D, A–E, variable number), domain, task, category, Rasch `b`, p-value, point-biserial, enemy, and arbitrary custom metadata. All mappings except selected text content are optional; create stable synthetic internal row IDs only if an ID column is absent.
- Metadata (including all unselected columns) is retained and available in result tables/visualization. Numeric vs categorical types recognized with an explicit user override.
- Reject dangerous or ambiguous mappings, missing required embed text, duplicate IDs with conflicting content, and malformed numeric labels with actionable messages. If identical duplicate records exist, deduplicate with a report; preserve original source-row index.
- Never arbitrarily truncate fields, silently impute original answers, fabricate correct options, or attach a default clinical domain.
- Provide normalized item previews that preserve **all** answer options and answer key flags if present.
- Handle whitespace, nulls, mixed Excel numeric/string IDs, long fields, and records lacking metadata. Escape rendered HTML/Markdown as needed to avoid code or HTML injection.
- If embedding text includes answer labels/options, disclose that this may increase surface lexical overlap and answer leakage. Default embed: vignette + stem + lead-in; option-text inclusion is an **explicit optional toggle**, not forced. Do not embed `rasch_b` or classification labels by default.

### Canonical records

```python
ItemRecord = {
  "internal_id": "stable-id",
  "source_row": 3,
  "item_id": "ITEM-103",
  "display_fields": {"stem": "...", "domain": "...", "rasch_b": 0.45},
  "metadata": {"domain": "...", "task": "...", "rasch_b": 0.45},
  "options": [
    {"option_id": "A", "text": "...", "is_key": False},
    {"option_id": "B", "text": "...", "is_key": True}
  ],
  "selected_embed_text": "Vignette: ...\nStem: ...",
  "source_fingerprint": "sha256..."
}
```

## 6. Embedding pipeline and in-memory index

1. Prompt the user to provide an OpenAI API key via password-masked UI field **or** use `st.secrets`/environment key. Never expose its value to the front end beyond entry; do not persist to disk, URL, exports or logs. Display a privacy notice **before API submission**.
2. On explicit **Generate Embeddings** action, batch nonempty item text; honor API limits and retries for rate limits/transient errors with bounded exponential backoff. Show processed/remaining, errors, skipped rows and requested embedding model.
3. Create `embeddings: np.ndarray[float32]` of shape N × d, record model ID, output dimensions, ordered record hashes and concatenation template. Normalize vectors by L2 with zero/NaN guards. Embedding vector for a new query must use the **same model and settings**.
4. Use a per-session hash or portable project manifest to avoid redundant paid calls. If text selection/model changes, invalidate corresponding embeddings and dependent results; warn before regeneration.
5. Enable CSV/JSON project export of metadata and a compressed NumPy `.npz`/portable bundle of vectors plus manifest. On import validate exact item order/hashes, shape, dimension and model identity; refuse mismatches and prompt regeneration.
6. For N < 1,000 and 1536-dimensional float32 vectors, approximate raw vector memory is under 6 MB. Embedding remote calls—not vector search—are likely the main cost/latency driver.

### Similarity math

For query vector **q** and candidate item vector **v_i**:

\[
\mathrm{cosine}(q,v_i)=\frac{q^\top v_i}{\|q\|_2\|v_i\|_2}.
\]

After unit normalization, score is `embeddings @ query_unit_vector`, and is theoretically in [-1, 1]. Do not rescale it without labeling the transformation. The threshold 0.85 is an **illustrative user-set threshold**, not a universal criterion for true sibling items. Show the candidate similarity distribution and counts across candidate thresholds; never convert cosine to an empirical probability of same-task status. Prefer label **Minimum cosine similarity** over **Top P**, which can be confused with the CTT item p-value.

## 7. Retrieval, rank modes, uniqueness and optional keyword search

**Primary/default:** exact cosine similarity. Support query by arbitrary input text, a selected existing item (same original embedding), or an item not in the uploaded bank with optional options. When querying by an existing item, exclude self-match by exact internal item ID. Always show the target separate from retrieved candidates.

Search filters:
- `Top K`: highest-scoring K **unique item IDs** (not K options; K defaults to 5; do not return > N).
- `Cosine threshold`: all unique items with `similarity >= threshold`, default example 0.85.
- `Both`: threshold candidates first, then top K by descending cosine; ties deterministic by ID.
- Filter by metadata where requested, with an explicit candidate-count display.
- If no candidates match, explain and allow lower threshold or broader candidate pool. Never pad with fabricated items.

**Optional secondary method:** BM25 keyword ranking. Implement using `rank_bm25` or a small, tested library with a sane tokenizer and configurable weighting; consider clinical abbreviations, units and hyphenation. Provide a separate toggle, not the default algorithm. Optional hybrid result uses clearly labeled weighted normalized combination **or RRF with separately generated ranks**, e.g. `score_RRF = Σ_j w_j / (k0 + rank_j)`; ensure RRF is identified as a ranking measure, not cosine similarity. For hybrid searches, **apply a cosine threshold to the original cosine column**, never to raw BM25 or RRF scores. Display columns for `cosine_similarity`, `bm25_score` (if applicable), and `hybrid_score` separately.

Every returned candidate shows ID, similarity, task/domain, full item stem, **all options**, correct key when provided, metadata, and selectable state. Support CSV/JSON exports.

## 8. Target task, sibling confirmation and cognitive interpretation

### Critical ordered sequence

1. Define **target item**: user-supplied/new text or selected bank item.
2. Retrieve distinct high-cosine candidate items.
3. Infer target task/domain using available text plus metadata. Show the **target task of the input item separately** from each candidate's apparent task. AI must flag ambiguities and cite exact supplied text fragments, not invent blueprint codes.
4. SME accepts, rejects or edits proposed task, and explicitly marks each candidate **same task / different task / uncertain**. Do not silently treat high similarity as confirmed same-task.
5. Only **accepted, distinct same-task candidates** comprise the default denominator for subsequent sibling feature frequency. Offer a visible optional switch for all retrieved candidates and record the switch in the report.
6. Extract and consolidate features from the target and accepted siblings; keep target support distinguishable. Obtain an auditable list of exact source item/option references per feature.

### Definitions (use exactly as conceptual rules)

- **Declarative key feature:** generalizable clinical/assessment fact, principle, diagnostic criterion, or knowledge required to solve the **target task** and obtain the item correct; not merely contextual wording.
- **Procedural key feature:** generalizable clinical or cognitive action, decision rule, sequence, prioritization, or updating step needed to perform the target task. May be **one dominant procedure** and/or **multiple valid candidate procedures**. Represent candidate relationship and ordering when applicable, and do not collapse distinct competing procedural strategies.
- **Instance feature:** scenario-specific person/context facts or presentation details that support the item but **are not intended to be the key difficulty-driving construct**; e.g. a particular age, blood-pressure reading, setting or presentation. Do not assume instance features are empirically difficulty-neutral: this is a design intention, testable with data.
- **Distractor feature:** conceptual misconception, incomplete knowledge, unsafe action, reasoning error, or misconception implied by a **specific incorrect option**. Do not infer an examinee's actual misconception solely from text. For each option, distinguish proposed misconception *mechanisms* from demonstrated examinee response processes. Multiple misconceptions may map to one distractor and multiple distractors may map to one feature; do not force a one-to-one model. Each distractor feature must link to option IDs and the associated item's key if available.

**Separate feature taxonomy/class labels from user-selected domain/task metadata.** The LLM should generate canonical semantic feature labels; SMEs retain final authority.

### Output schema (structured output, versioned)

Use a Pydantic-validated JSON schema similar to:

```json
{
  "schema_version": "1.0",
  "target_item_id": "TARGET-01",
  "target_task": {
    "name": "Prioritize initial intervention for suspected sepsis",
    "domain": "Acute care",
    "decision_or_action": "Select most urgent nursing action",
    "rationale": "Based on the lead-in requiring first action",
    "source_item_ids": ["TARGET-01"],
    "evidence_spans": [{"item_id": "TARGET-01", "field": "stem", "excerpt": "..."}]
  },
  "candidate_task_alignment": [
    {"item_id": "ITEM-01", "proposed_alignment": "same_task", "evidence": "...", "confidence_status": "uncertain", "sme_decision": "unreviewed"}
  ],
  "features": [
    {
      "feature_id": "F001",
      "canonical_label": "Recognize evidence of impaired perfusion",
      "definition": "...",
      "feature_type": "declarative_key",
      "task_relevance": "...",
      "dominance": null,
      "procedure_group": null,
      "supporting_items": ["ITEM-01", "ITEM-02"],
      "source_spans": [
        {"item_id": "ITEM-01", "field": "stem", "option_id": null, "excerpt": "...", "evidence_type": "explicit_text"}
      ],
      "inference_rationale": "If a required concept is not stated literally, explain the task-to-feature inference here.",
      "proposed_misconception": null,
      "llm_uncertainty_note": "",
      "review_status": "pending",
      "sme_importance": null,
      "sme_note": ""
    }
  ]
}
```

Feature types must be enumerated as `declarative_key | procedural_key | instance_context | distractor_feature`. `dominance` for procedures: `dominant | alternative | contingent | unknown` as applicable. No fictional citations. Verify cited item IDs and option IDs actually exist in the payload. **Evidence types:** `explicit_text` requires an exact source substring; `clinically_inferred` requires an item/option reference plus a transparent reasoning rationale, not a fabricated quotation. Mark unsupported proposals for SME adjudication rather than silently treating them as observed. For strictly validated JSON Schema, use compatible required keys/nullable fields and `additionalProperties: false` as supported by the selected API; handle refusal, truncation, and schema errors explicitly.

### LLM prompting constraints

**System instruction for task extraction** (save as reusable prompt asset):

> You are an assessment-design specialist. Identify the main target task measured by the lead-in and response requirement. Distinguish task from content domain. Use only the provided items and metadata; cite exact supporting excerpts. Treat high cosine similarity as retrieval evidence, not proof of equal tasks. Return JSON matching the required schema. Explicitly mark uncertain judgments.

**System instruction for feature decomposition**:

> You are a psychometrician working with a clinical SME on a key-feature cognitive model. For the target task and SME-confirmed sibling items, extract all distinct, evidence-supported declarative key features, procedural key features (dominant and/or alternative), instance/context features, and distractor features. A key feature is essential to solving the item; an instance feature supplies contextual manifestation not intended as a main difficulty driver. Link every feature to source item and, where relevant, option IDs. Avoid duplicates by canonical meaning; preserve genuine distinct actions or concepts. Do not impose any fixed feature count. Do not invent source statements, infer response behavior, or declare distractor misconceptions as facts. Return only schema-valid JSON for SME review.

**Canonicalization pass**:

> Normalize semantically equivalent candidate features into a unique label *within the same conceptual category*, preserving the union of supporting item references. Never merge contradictory claims or distinct procedural sequences; instead create related-feature links. Ensure that identical source item IDs are counted once per canonical feature. Show uncertainty when the merge is contestable.

Supply the task, accepted item IDs, and actual item/option excerpts in the user payload. In the app, segment LLM tasks into bounded requests if prompt budget is exceeded; provide intermediate validation and deterministic canonicalization. Never assume the LLM has seen all uploaded rows unless sent. Use response schemas rather than parser-dependent free text. Validate every output before display.

## 9. Feature-space uniqueness, frequency, and SME workflow

### Unique feature semantics

- Canonical feature labels are **unique within category**; deduplicate exact and strongly equivalent paraphrases while keeping a provenance mapping. Across categories, labels may be similar but indicate substantively distinct roles; do not silently merge categories.
- **All answer options** of each unique sibling item appear in the distractor review; only incorrect options contribute distractor features. If the answer key is missing or ambiguous, display all options but leave key-dependent analysis pending.
- For two options in the same item that evidence one conceptual distractor feature, that feature's frequency increments **only once** for that item.
- A feature may appear in the target item and/or retrieved items; display target support separately from accepted-sibling frequency.

For N confirmed **unique** sibling items and feature f, define the incidence matrix:

\[
A_{if}=\mathbf{1}\{\text{item }i\text{ contains canonical feature }f\},\quad
n_f=\sum_{i=1}^{N}A_{if},\quad
p_f=100n_f/N\%.
\]

**Required example:** If five distinct same-task sibling items were accepted and a key feature is present in all five, display **Frequency = 5**, **Coverage = 5/5 = 100%**; if present in three, show **3/5 = 60%**. The target input item is *not* in N by default. With N = 0, show N/A and a meaningful message, never divide by zero. Feature frequency is an **observed annotation of the current retrieved set**, not a population prevalence estimate or evidence that the feature increases difficulty. Preserve both raw extracted incidence and SME-approved incidence; report which is shown. In distractor space, optionally show a separate **option count** alongside item frequency, labeling each denominator clearly. Do not infer support based solely on lexical overlap if SME judges the underlying construct different.

### SME interaction

- Display four tabs/sections: **Declarative**, **Procedural**, **Instance**, **Distractor**; show **all** unique proposed features, no arbitrary top-five truncation.
- Each row: canonical label; definition; supporting item IDs/options; frequency; percent; proposed task relationship; explicit/inferred evidence flag; dropdown importance rating **1–5**, with `unrated` default; accept/reject/edit controls; SME note; review state. Define anchors: 1 = peripheral, 2 = low, 3 = moderate, 4 = high, 5 = essential. Ratings are not forced to sum to a fixed total, and the number of extracted features is not fixed. Track `reviewer_id` or pseudonym, timestamp, schema/model version, edit history, and original AI suggestion without storing credentials.
- 1 = minimally important; 5 = essential. The **number of features** and the **rating scale** are independent. If needed, make the rating scale configurable per study, but default 1–5.
- For procedural features, allow one dominant procedure and multiple alternatives/candidates; do not force mutually exclusive choice if multiple procedures are genuinely required. Provide `dominance` selector.
- Preserve raw LLM proposals and SME-approved canonical versions in session data and exports. Never overwrite SME changes on rerun, re-search or LLM retry without confirmation. Review actions must propagate to feature matrices/graphs and exports.
- Provide accepted-only, all-proposed, and rejected views, with a visible denominator based on confirmed sibling selection.

## 10. Visualizations

### Item embedding plot (PCA, t-SNE, optional UMAP)

- Use Plotly scatterplot in 2D; one point **per unique item**. Show PCA as fast default, t-SNE optional, UMAP optional. Optional 3D plot only if easy and stable.
- User selects a metadata column (`domain`, `task`, `type`, `rasch_b`, etc.) to color points. **Categorical** → distinct palette and legend; **numeric** → continuous scale/legend. Preserve source values including missing/unknown category.
- Hover tooltip displays item ID, selected metadata, shortened stem, and relevant score; click or nearby row selection reveals full item and options. Support hiding/filtering categories, highlighting target vs accepted siblings vs rejected candidates, and exported plot data.
- Explain **2D distances are projections, not cosine similarity values**. Rank/retrieve in original embedding space only. Do not compute search ranks from 2D coordinates.
- Deterministic seed. For t-SNE ensure perplexity < sample count; disable gracefully for extremely small samples; consider PCA to 20–50 dimensions when helpful. Warn that t-SNE/UMAP global distances and cluster shape may be unstable.
- Refresh plot when metadata color/filter changes without calling embedding API again.

### Interpretable cognitive-feature plots

- Unique item × canonical feature **binary incidence heatmap**, grouped by declarative/procedural/instance/distractor categories, with support counts, item IDs, and SME annotations.
- Optional bipartite item-feature network; feature size may encode frequency and color feature category. Show target and sibling item links.
- Optional 2D embedding plot of **feature definitions** if separately embedded (explicitly label it as feature-embedding space, not an empirical latent dimension); no causal or IRT interpretation unless separately modeled.
- Display provenance on hover/click and simple downloadable matrix `item_id, feature_id, feature_type, value` for reproducibility.

## 11. Heuristic psychometric analytics (do not oversell)

### Difficulty

Only when uploaded item bank includes an empirical difficulty parameter. Default target metric = user-mapped Rasch `b`; do not silently substitute classical p-value. **Before prediction, require explicit confirmation that difficulty parameters are comparable on a common calibrated/linked scale and have consistent direction and units.** If this is unknown, display descriptive neighbors but label the aggregate as exploratory/non-comparable and disable any purported calibrated predictor. Do not pool Rasch b, 2PL b, percentage-correct, or transformed scales without an explicit justified conversion. For neighbors with observed difficulty `b_i` and cosine `s_i`, report:

\[
\widehat{b}_q = \frac{\sum_{i\in\mathcal{N}} w_i b_i}{\sum_{i\in\mathcal{N}} w_i},\qquad w_i=\max(s_i,0)^\alpha
\]

with configurable `alpha` (default 1); also report unweighted mean, SD, min/max, N, label units, and neighbor table. If sum(weights) <= 0 or too few labeled neighbors, abstain. Provide at least a naive mean-difficulty baseline and, when sample size supports it, leave-one-out *and grouped-by-stimulus/sibling-family* holdout evaluation to detect near-duplicate leakage; report MAE/RMSE/bias, number of evaluated items and per-domain error if feasible. Do not evaluate training and test observations with overlapping duplicate/linked-case families. Estimate uncertainty using validated residual/error diagnostics only; do not manufacture model-based confidence intervals from neighbor SD. Do **not** claim the estimate is an IRT calibration or provide unsupported confidence intervals. Recommend empirical validation before operational use.

### Classification

For a mapped categorical field, similarity-weighted voting among labeled distinct candidates, weighted category support, N, and optional abstain on tie/low evidence. Only compare categories with a confirmed shared taxonomy; handle absent, hierarchical, and multi-label categories explicitly rather than forcing single-label majority when inappropriate. Keep domain and target task separate. If classification metadata was embedded, warn about label leakage and offer exclusion. Holdout diagnostics: confusion matrix and per-class accuracy when enough labeled items exist. The LLM may propose a task but should not overwrite provided validated labels. Show misclassifications, macro-F1/balanced accuracy when feasible, and class counts, not only raw accuracy.

### Enemy detection

Flag review candidates based on exact duplicate/near-duplicate stem, high cosine, option overlap, shared precise case/data, common key facts, and/or existing `enemy` flags; provide independent flags and evidence. Differentiate intentional linked-item sets from prohibitive enemy pairs; distinguish semantic similarity, cueing/answer disclosure, shared stimulus dependence, and item-security concerns. Record symmetric pair IDs, rule triggers and SME decision. Keep rule-based thresholds editable; explicitly treat all flags as **candidate enemies**, not validated true enemies. Never infer common exposure/test-security risk from similarity alone.

## 12. Interface requirements (Streamlit)

Create a polished, responsive app with sidebar and main workflow:

- Sidebar: **Data & API**, **Search**, **Decomposition**, **Distractors**, **Embedding Visualization**, **Difficulty**, **Classification**, **Enemy Candidates**, **Exports**. Use either `st.tabs` or a small navigation menu; avoid losing state when navigating.
- Data pane: drag-drop upload, multi-sheet choice, schema auto-detection, all-column preview, map input text and metadata, ID/key/options, stats and warning banners; explicit **Build Embeddings** button.
- Search pane: target item from uploaded bank or pasted text, cosine-first, Top K/threshold/both, optional keyword toggle, clear result counts, deduplicated table and complete option cards; submit button.
- Task pane: task suggestion with source evidence, edit/confirm, and each sibling candidate checkbox/tri-state status before feature extraction.
- Feature pane: no feature count limit; four categorized panels, feature frequency with denominator, supporting IDs, multi-option distractor references, evidence text, dropdown importance rating 1–5, status/edit tools, procedural dominant/alternative selector.
- Visualization: PCA/t-SNE/UMAP selector, metadata color dropdown, category filters, target/sibling highlighting, hover/click details, item-feature heatmap, optional network.
- Status dashboard: file name, N items, N embedded, embedding model/dims, API readiness, N retrieved, N accepted, N features, percentage SME-reviewed; clear warnings about ephemeral storage and API cost.
- Export center: results CSV/JSON, reviewed feature mapping CSV/XLSX, item-feature incidence CSV, dataset/embedding project bundle, plot export when practical.
- Use keyboard-accessible controls, readable color palettes, color-independent markers for target vs siblings, and functional empty states.

## 13. Security, privacy and reliability requirements

- OpenAI API keys must never be committed, logged, displayed in diagnostics, copied into reports, or written to saved archives. Use `st.secrets`, env vars, or session-only masked input; secure precedence and help text.
- API requests must occur **server-side** only, on explicit user action and consent; explain that item content will be transmitted to OpenAI for embeddings and/or decomposition. Data may be proprietary; permit a user to back out and continue local metadata exploration without sending content.
- Treat uploaded text and LLM outputs as untrusted. Guard against prompt-injection instructions inside item stems, HTML injection, CSV formula injection on exports, and giant/invalid uploads. Prompt-injection text should be data, never task instructions. Include a consent toggle for external calls.
- Build fail-soft behavior: blank/missing keys, zero vectors, partial API failures, malformed schemas, label/ID mismatches, t-SNE tiny N, empty sibling set, missing Rasch labels, unsupported Excel files, upload size constraints.
- Provide progress feedback and avoid repeated paid calls on Streamlit reruns. Optional retrieval and viewing should work offline **after** vectors have been generated/imported.
- Warn before changing dataset, mapping or embedding model because downstream analyses become stale; provide clear reset/redo controls.
- Do not ship a fixed simulated NCLEX dataset; a tiny separately created synthetic test fixture is fine and must be clearly marked demonstration-only.

## 13A. Required cross-role validation and human-review safeguards (v2)

**Developer review:** inspect upload parsing and session isolation; verify app runs with zero data and zero API key; verify retry/backoff, per-row embeddings, fingerprint invalidation, accurate API-cost status, complete outputs, and no global cache content bleed. Maintain tests for no information lost from variable option counts; no silent partial writes; 1,000-item vector processing within a modest memory budget. Always display an explicit *not yet implemented* status if the LLM or plots cannot be run without an API key.

**Psychometrician review:** verify that target task is a cognitive demand (decision/action), not just a content label. Audit the feature incidence matrix at item level, not repeated phrase count. Require sample same-task and different-task pairs across multiple cosine bands rather than assuming 0.85 means siblings. Validate difficulty-scale comparability, perform grouped holdout and naive baseline for heuristics, identify key/option-label leakage, distinguish frequency from effect size, and ensure projected 2D distances are never used as similarity evidence. Do not label features as scientifically validated or diagnostic misconceptions solely because an LLM generated them.

**Item writer / SME review:** require a side-by-side comparison of target and candidate items: scenario, lead-in, options, keyed answer (if present), intended task and domain, AI rationale, source excerpts, and controls *same task / different task / uncertain*. The SME must approve the task before accepting extracted key-feature spaces. Features can be explicitly present or legitimately inferred through the task; an inference needs its reasoning and source-item anchors. SMEs can mark a proposed feature as not required, context-only, ambiguous or unsupported. Provide edit/merge/split workflows with canonical IDs preserved and audit history. Include one dominant procedural strategy or multiple contingent/alternative strategies when appropriate; display sequence/condition, not a flat unordered list. Treat distractor misconception labels as hypotheses requiring SME verification. If item keys/clinical rationales are absent, do not silently manufacture them.

**No external review fiction:** The autonomous agent may perform a *simulated role-based inspection* using these checklists; it must not describe that as approval by an actual human psychometrician or clinical SME.

## 14. Automated test suite and concrete acceptance tests

Write real `pytest` unit tests, API-mocked integration tests, and at least one Streamlit smoke test (use `streamlit.testing.v1.AppTest` if suitable). Tests must not depend on live OpenAI API or confidential user data.

**Mandatory checks:**

1. CSV and XLSX import with 2, 4, 5+ option columns; mixed IDs and types; correct auto-detection and manual mapping. Duplicate conflicting IDs trigger clear error; exact duplicate rows are reported and deduplicated.
2. Selected-column concatenation reproducible; unselected metadata preserved; changing text columns invalidates vectors; changing color metadata does not.
3. Embedding mock produces N vectors with declared dimension; missing/invalid embeddings are handled; API failure doesn't corrupt existing session state; reimport mismatch is detected.
4. Cosine search returns mathematically expected ordering (test against hand-computed normalized vectors), self-match exclusion, negative/zero handling, Top K, threshold 0.85, both, and stable ties.
5. **Uniqueness:** K=5 returns at most five distinct item IDs and preserves all options; an item with five distractors does not consume five result slots.
6. If cosine 0.86 and BM25 is poor, the item still appears in primary semantic results; threshold 0.85 is never applied to raw BM25/RRF.
7. Task confirmation changes accepted sibling set; high-similarity but different-task items are not silently included in the feature denominator.
8. LLM parser validates allowed feature types, item IDs, option IDs, evidence excerpts and sources; unsupported claims are flagged; no fictional IDs.
9. All four feature types can have arbitrary counts; deduplication merges equivalent features within a type while preserving provenance; procedural alternatives remain distinct when appropriate.
10. Five accepted distinct sibling IDs with an identical canonical feature produce exactly `count=5`, `denominator=5`, `coverage=100%`; a feature in 3 produces `count=3`, `coverage=60%`; repeating the feature 3 times within one item still counts only 1; target item does not inflate denominator.
11. SME accept/reject/edit, importance 1–5, and dominant-procedure selection persist across Streamlit reruns and exports; LLM retry does not overwrite approved edits.
12. Metadata categorical and numeric color controls operate without recalculating embeddings; visualization gracefully handles tiny samples.
13. Difficulty and classification disable politely if labels are missing; do not conflate Rasch `b` with p-value; note predictive limitations.
14. Enemy detector reports candidate flags with evidence but doesn't auto-label official enemy pairs.
15. Codebase contains **no hard-coded credential**; project bundle excludes keys; prompt injection inside a stem is not treated as an instruction.
16. `streamlit run app.py` succeeds; static/syntax checks, `pytest -q`, and a synthetic end-to-end run succeed. Provide a simple deployment README.
17. Two simulated independent users load disjoint confidential files: no uploaded records/embeddings/review text/keys appear in the other's state or shared cache. Session reset demonstrates expected ephemeral behavior and a functioning export/reimport path.
18. Mocked LLM yields (a) explicitly supported item feature and (b) valid clinically inferred but unstated knowledge. The schema preserves their different evidence types; fabricated quoted excerpts are rejected; unsupported inferences are routed to SME review.
19. Five retrieved items include one high-cosine different-task item; only four SME-confirmed siblings enter the default denominator. Target item is excluded even if present in the same bank.
20. Difficulty labels from two artificially unlinked Rasch scales cannot be averaged as calibrated item difficulty; when scale comparability is confirmed, compare weighted kNN versus a naive baseline using grouped held-out families.
21. Mixed labeled categories, multi-label tags, no labels, or incorrect/blank item keys do not create an invented key, forced classification or unsupported distractor diagnosis.
22. Option-level support and item-level frequency differ: three distractor options in one item mapping to one canonical misconception give option count 3 but item frequency 1; reports label these separately.
23. SME edits, merges, splits, review states and procedural ordering survive Streamlit reruns and project export/reimport; re-embedding invalidates dependent results with warnings but never silently overwrites approvals.
24. API consent is required before network calls; mock mode performs an end-to-end demo without any external requests and reports estimated cost/usage and error state without secrets.
25. Similarity cutoff 0.85 is labeled a chosen heuristic, not a probability. Retrieval ranks use original embedding space even when t-SNE or UMAP projected coordinates are present.
26. A candidate item intentionally linked in a multi-part stimulus is not automatically declared an enemy; evidence and human approval required.

### Suggested synthetic demonstration

Make an **independently invented** 8–12-item bank with item IDs, several clinical domains, variable option counts, domain/task metadata, some observed Rasch difficulty values, and purposely paraphrased/similar stems. Use it for UI smoke tests and frequency test scenarios; don't rely on it being deployed by default. Include a hand-constructed embedding-matrix fixture for deterministic threshold tests.

## 15. Implementation phases and iterative role-based review

Execute in order, reporting artifacts/tests after each. Do not stop after scaffolding.

1. **Source audit and plan** — map old modules and identify useful components vs hard-coded NCLEX/SQL assumptions; document security risks. Reviewer: engineer + security.
2. **Ingest/schema and state** — implement robust file upload, mapping, normalized item records, deduplication, editable preview; tests. Reviewer: developer + SME.
3. **Embeddings and search** — OpenAI adapter, consent, batch/cost handling, in-memory vectors, cosine-first search, optional BM25/RRF, exports; tests. Reviewer: engineer + psychometrician.
4. **Task alignment and SME decisions** — structured task extraction, accepted sibling set; tests. Reviewer: psychometrician + SME.
5. **Cognitive decomposition and feature frequencies** — four types, procedural dominance, per-item sources, canonicalization, no fixed limit, SME ratings, audits; tests. Reviewer: all three core roles.
6. **Distractors, predictions, enemies** — full options, review and heuristic analytics with qualifications; tests. Reviewer: psychometrician + SME.
7. **Visualizations and UX** — PCA/t-SNE/UMAP and metadata colors, feature heatmap and interaction; visual QA. Reviewer: developer + SME.
8. **Security/reliability, documentation, deployment readiness** — API secrecy, synthetic fixtures, data privacy, packaging, Streamlit startup, smoke tests, README, final role-based review and fixes.

### Review rubric

Each reviewer must assess **correctness, evidence/provenance, generality to arbitrary imported columns, duplicate handling, item-writing construct validity, SME control, model/provider privacy, cost and Streamlit Cloud practicality**. Revisit code and tests until high-severity issues are fixed. Avoid claiming psychometric validation for unvalidated heuristics or clinical correctness for unreviewed LLM outputs.

## 16. Definition of done and output format for executor

**Done means:** A runnable Streamlit app satisfying all mandatory acceptance tests, complete setup and run instructions, a clear API key/consent flow, a database-free demonstration that works with any conforming CSV/XLSX bank under 1,000 items, and exports for repeat use.

At completion, report in this order:

1. **Repository location and exact files created/edited**, with links or relative paths.
2. **Implemented user flows**, identifying all 12 major capabilities and what is truly operational.
3. **Architecture decisions**, especially in-memory vectors, dynamic column mapping, OpenAI embedding/chat adapters, cosine-first design, and candidate SME validation.
4. **Testing evidence:** commands, passed/failed counts, important edge cases, manual UI checks.
5. **Launch instructions:** setup, environment, `streamlit run app.py`, and Streamlit Cloud secrets configuration (no real secrets).
6. **Remaining limitations:** missing OpenAI key/test data, external permission, known defects, unvalidated psychometric assumptions, and remaining enhancements.

Do not state that work is completed until the app and test suite actually run. If environment tools cannot execute, provide exact blocker and build as much code as possible rather than inventing test success.

## 17. References and implementation checkpoints

Verify current SDK signatures before coding; avoid pinning to a deprecated API merely to match old source code.

- OpenAI Embeddings guide: https://developers.openai.com/api/docs/guides/embeddings — `text-embedding-3-small` default 1536 output dims.
- OpenAI Embeddings API reference: https://developers.openai.com/api/reference/resources/embeddings/methods/create
- Streamlit session/caching state (WebSocket reset and cross-user global cache caveat): https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state and https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
- Streamlit secrets management: https://docs.streamlit.io/develop/concepts/connections/secrets-management
- OpenAI structured model outputs: https://developers.openai.com/api/docs/guides/structured-outputs — verify chosen model/schema support (including strict object restrictions) before implementation.
- Scikit-learn t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html — perplexity must be less than sample count.

**Final reminder:** This app is a **proof of concept for SME-assisted item analysis and cognitive decomposition**. It must prioritize **auditability, uniqueness, per-item evidence, human control, and valid mathematical computation** over an impressive-looking interface or unverified AI claims.
