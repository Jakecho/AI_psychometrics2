# Item Bank Analytics & Cognitive Feature Decomposition (POC)

A database-free Streamlit proof of concept for SME-assisted item analysis. Upload an item bank (CSV/XLSX, up to 1,000 items), embed it with OpenAI (or an offline mock), retrieve similar items by exact cosine similarity, confirm which candidates measure the same task, decompose the task into declarative, procedural, instance, and distractor features with traceable evidence, and review everything as an SME.

Implements [`item dev 2/ITEM_ANALYTICS_MASTER_PROMPT_REVIEWED_v2.md`](../item%20dev%202/ITEM_ANALYTICS_MASTER_PROMPT_REVIEWED_v2.md).

Workflow: **Upload → Map columns → Embed → Search → Confirm same task → Decompose → SME review → Visualize / Export**.

In **Decomposition → Step 4**, SMEs can add elements to each space (key features, instance features, distractor candidates) with verified evidence, and remove or restore elements. **Feature Spaces** lists every space in the session and filters it by task or domain.

**Space Library → Database** keeps reviewed spaces across sessions in a dedicated SQLite file (`data/space_library.sqlite` locally). Save a space from the end of the Decomposition review or from **Space Library → Session spaces**. You can then search by workspace, task, domain, space type, status or keyword; edit elements in place; remove or restore elements; delete, restore or purge spaces; review the history; and back up or restore the library as JSON. Saves carry a version number, so a stale copy cannot overwrite newer edits.

**Import** has three tabs: **Bank import** (the item bank), **New items** (items to compare against the bank, typed or from an Excel/CSV file such as `samples/synthetic_queries.xlsx`), and **API & embeddings**. New items are embedded with the bank's model when added and are never added to the bank. In **Search**, pick a bank item or a new item from its dropdown, or click **Search all new items** for a downloadable batch table.

## Run locally

```bash
cd item-analytics-poc
python -m pip install -r requirements.txt
streamlit run app.py
```

In **Import → Bank import**, click **Load synthetic demo bank** to try the whole workflow in **Offline mock** mode. This mode makes no network requests. Its vectors are lexical hashes and its proposals are deterministic heuristics, labelled as such. Mock output is not AI or SME analysis.

For real analysis, choose **OpenAI** mode and provide a key in one of three ways: the masked session field, `OPENAI_API_KEY` in `.streamlit/secrets.toml`, or the environment. Then enter the model ID for task and feature analysis (it must support structured JSON-schema output) and tick the two consent boxes. One consent covers embeddings and the other covers analysis.

## Deploy on Streamlit Community Cloud

| Setting | Value |
| --- | --- |
| Main file path | `item-analytics-poc/app.py` |
| Python | 3.11–3.13 |
| Secrets | `OPENAI_API_KEY = "..."` (optional; users can also paste a key per session) |
| Space database | Off by default when hosted. To enable it, set `ITEM_ANALYTICS_DB_PATH` to a writable path. Everyone with access to the app then shares the library, and Community Cloud storage may not survive a redeploy, so keep JSON backups. |

Storage is session-only. A refresh or disconnect loses unsaved work, so use **Exports → Project bundle** and re-import it from **Data & API**.

## Tests

```bash
cd item-analytics-poc
python -m pytest -q          # 81 tests; all external network access is blocked during tests
```

The suite covers the spec's acceptance checks with synthetic data. It uses mocked OpenAI clients and Streamlit `AppTest` for an end-to-end run, section rendering with no data and no key, two-user isolation, the consent gate, and edit persistence across reruns.

## Capability status

| # | Capability | Status |
| --- | --- | --- |
| 1 | Import and prepare (CSV/XLSX, sheet choice, profiling, mapping, dedup, variable options, key parsing) | Implemented |
| 2 | Embeddings (OpenAI SDK, batching, bounded retries, reuse by text hash, retry failed rows, bundle) | Implemented; verified against a mocked client and SDK 2.26 signatures. No live call was made. |
| 3 | Cosine retrieval (exact, Top K / minimum cosine / both, self-exclusion, metadata filter, BM25, RRF; bank item or new item (typed or CSV/XLSX) from a dropdown, plus batch search over all new items) | Implemented |
| 4 | Task validation (AI proposal with verified evidence, SME accept/edit/reject, same/different/uncertain per candidate) | Implemented |
| 5 | Four-space decomposition (explicit vs inferred evidence, no count limit, dominance/order, edit/merge/split, 1–5 rating, SME add and reversible remove/restore of elements) | Implemented. Canonicalization is deterministic (exact normalized labels); semantic merges are made with the SME merge tool. |
| 6 | Distractor review (all options, key status, option-level links, misconception hypotheses, SME-added candidates tied to incorrect options) | Implemented |
| 7 | Feature frequencies (once per distinct confirmed sibling, N/A at N = 0, SME vs raw incidence, option count) | Implemented |
| 8 | Visualization (PCA/t-SNE/UMAP, metadata colour, category filter, role markers, click-through, incidence heatmap) | Implemented. Optional item-feature network graph and feature-definition embedding plot are not implemented (P2). |
| 9 | Difficulty heuristic (scale confirmation, unlinked-scale refusal, weighted kNN, grouped holdout vs naive baseline) | Implemented |
| 10 | Classification heuristic (weighted vote, abstention, multi-label, leakage warning, SME decision, grouped holdout metrics) | Implemented |
| 11 | Enemy candidates (independent triggers with evidence, linked sets kept separate, SME decisions) | Implemented |
| 12 | Exports (CSV/XLSX/JSON, incidence matrix, evidence map, audit history, project bundle without keys) | Implemented |
| 13 | Feature-space library (current and archived spaces, searchable by task, domain, space type, status and keyword) | Implemented |
| 14 | Space database (SQLite): save reviewed spaces; search, update, remove/restore, delete/restore/purge, history, JSON backup; version conflict protection | Implemented |

## Design decisions

- **In-memory vectors.** An `N × d` float32 matrix of unit vectors; for N ≤ 1,000 at 1,536 dimensions this is under 6 MB. The dimension comes from the API output and is checked for consistency.
- **Cosine first.** All ranking uses `vectors @ q` in the original space. BM25 and RRF are optional, separately labelled columns. Thresholds apply only to cosine. 2D projections are display-only.
- **Dynamic mapping.** No fixed schema. Any columns can be embedded, any number of option columns can be mapped, and all other columns stay as metadata. The key and labels are never embedded by default.
- **Mockable adapters.** OpenAI is imported lazily behind `OpenAIEmbedder` and `OpenAIStructuredLLM` (Responses API, strict JSON schema, `store: false`). Both refuse to run without consent.
- **Evidence you can verify.** `explicit_text` spans must quote the cited field or option verbatim. `clinically_inferred` spans need an item reference and a reasoning rationale, and any quote they give must be real. Unverifiable support is removed and the feature is routed to SME adjudication.
- **The SME controls the result.** Each feature keeps an immutable copy of the original AI proposal. An SME-touched feature is never overwritten: new evidence from a rerun waits for explicit confirmation. A new search, changed mapping or re-embedding archives earlier reviews and never deletes them.
- **Layout.** Each section has a two-column main panel: controls and workflow on the left, results and review on the right, under a one-line status bar.
- **Isolation.** All user data lives in `st.session_state`. No `st.cache_data` or `st.cache_resource` is used. Projections are cached per session, keyed by embedding fingerprint, method, parameters and seed.

## Simulated role-based review (not a human review)

Each phase was checked from the developer, psychometrician and item-writing SME perspectives against the specification's checklists. These are an AI agent's simulated reviews, not sign-off by a real psychometrician or clinical SME.

| ID | Sev. | Area | Finding | Resolution |
| --- | --- | --- | --- | --- |
| R1 | P0 | app.py | Consent, mode and key widgets lost their state when the user navigated to another section (Streamlit widget cleanup). | Values are persisted in session state at the start of every run. |
| R2 | P0 | app.py | Feature-table edits were tracked by row position, so after a filtered view changed, an edit could reapply to a different feature. | Rows are matched by feature ID (`editor_changes`, tested), and the editor key is renewed after every applied edit. |
| R3 | P1 | app.py | The mapping form shared its key with the session's mapping state, which crashed on load. | Renamed. Found by the AppTest end-to-end test. |
| R4 | P1 | app.py | A selectbox display function read session state, which is fragile outside a script run. | Labels are precomputed. |
| R5 | P1 | tests | Holdout fixtures were near-duplicates within each class, so grouping (correctly) removed every same-class neighbour. | Fixtures were made distinct; grouping behaviour is unchanged. |
| R6 | P1 | psychometrics | Exploratory difficulty output could be mistaken for an estimate. | No estimate is shown until scale comparability is confirmed; multiple scale groups are refused. |

| R7 | P1 | space_db | Two edits to the same saved space in one table submission: the second was refused as a version conflict. | Versions are chained across edits (`apply_editor_changes`, tested). |
| R8 | P1 | psychometrics | Saved spaces mix SME-approved elements with unreviewed AI proposals. | Status is stored per element and the database view has a Status filter, so approved-only is one click. |
| R9 | P1 | security | The shared library stores quoted item evidence. | It stays off on hosted deployments unless configured. The UI says it is shared and holds item evidence. Tests use temporary database files. |

No P0 issue is known to be open.

## Remaining limitations

- No live OpenAI request has been made from this environment. Verify the chosen model supports strict structured output before relying on it.
- Mock mode demonstrates the mechanics only. Its features (for example "Presentation value: 118 bpm") are heuristic placeholders.
- Cost figures are rough character-based estimates using approximate list prices.
- The app does not yet provide a guided audit sample of same-task and different-task pairs across cosine bands. The threshold distribution table supports that review, but the sampling itself is manual.
- The difficulty and classification heuristics are not validated predictors. Use them for exploration until they are validated empirically.
- Legacy apps in this repository (`NCLEX_RN_VectorDB_v2.py`, `NCLEX_RN_VectorDB_v3.py`) contain a hard-coded database password. Rotate that credential; nothing from them was copied here.
