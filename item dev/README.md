# Family Medicine Item Development

A Streamlit workspace for subject-matter experts (SMEs) developing clinical assessment items, with separate local and Community Cloud entrypoints. The product specification is in [MASTER_PROMPT.md](MASTER_PROMPT.md). It is designed around three authoring modes, with a shared clinical model, review workflow, and optional AI assistance.

| Mode | Intended writing task | Response formats |
| --- | --- | --- |
| Standalone MCQ | Write a focused clinical question, or several items sharing one unchanged vignette. | Single-best-answer and multiple-selection |
| Unfolding Case | Develop linked decisions as new clinical information becomes available. Item count and disclosure-stage count are independent. | Either format for each item |
| Clinical Appraisal MCQ | Ask learners to evaluate a preliminary assessment, diagnostic recommendation, management plan, or referral decision. | Single-best-answer and multiple-selection |

This is an authoring tool. Its learner view is a preview of authored content, not a validated operational examination platform.

## Start locally

The existing parent project environment contains Streamlit 1.66.1 and pandas 2.3.3. From this folder, run:

```bash
./run.command
```

Or double-click `run.command` in Finder. The launcher first checks the parent project's `.venv/bin/python`, then `python3`, allowing up to 10 seconds per candidate to import Streamlit and pandas. It selects a healthy environment or prints a setup message. `ITEMDEV_PYTHON` overrides automatic selection. It installs nothing automatically and does not change the parent project. Open [the local workspace](http://127.0.0.1:8526) after the server starts. Keep the terminal running and press Control-C to stop it.

To create an independent environment instead:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ITEMDEV_PYTHON="$PWD/.venv/bin/python" ./run.command
```

The default server listens only on `127.0.0.1`; it is not published or shared with other computers. Do not expose this local authoring server publicly without a separate access-control and deployment design.

## Streamlit Community Cloud

Use the hosted entrypoint when deploying this application from GitHub:

| Deployment setting | Value |
| --- | --- |
| Repository | `Jakecho/AI_psychometrics2` |
| Branch | `main` |
| Main file path | `deploy/item_studio/streamlit_app.py` |
| Python version | `3.13`, matching the verified local environment |
| Dependencies | `item dev/requirements.txt`, alongside the hosted entrypoint |

In the [Community Cloud workspace](https://share.streamlit.io), choose **Create app**, select the repository/branch/main-file values above, choose Python 3.13 in **Advanced settings**, and deploy. Enter the path manually if it is not suggested. Streamlit documents the [deployment process](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [requirements-file placement](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

The cloud entrypoint uses isolated, session-only draft storage. It does not persist all visitors' work to a shared `data/library.json` file. Export authored cases before a browser reload, session disconnect, app restart, or redeployment; hosted storage is not a durable database. Reimport an exported case to continue later. Local `app.py` retains the durable JSON library described below.

AI remains optional. To enable it on the hosted app, add the supported `ITEMDEV_AI_*` and optional `ITEMDEV_TRANSCRIBE_*` settings as top-level entries in Streamlit's **Secrets** field, using the variables documented below. `cloud_entry.py` maps only its supported allowlist into the server environment. Leave them unset to use manual mode. Store actual credentials only in app settings, never in GitHub. See [Community Cloud secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

A cloud server cannot reach an Ollama service running only on your personal computer's loopback address. For hosted AI, configure a service reachable from the deployment environment; configuring its URL does not verify connectivity or model quality.

## Authoring workflow

1. Choose the authoring mode and begin with a blank draft or an educational example.
2. Enter a clinical idea, shared vignette, blueprint domain, subdomain, and objective.
3. Choose the targeted reasoning phase or appraisal dimension. For unfolding cases, define stages, disclosure, and transitions separately from the number of items.
4. Write each lead-in, options, key or keyed option set, and option-level rationales. Link tempting alternatives to proposed misconceptions and the case evidence that distinguishes them.
5. Run structural validation and inspect issues. An incomplete draft may still be saved for later work; a passing structural check is not clinical approval.
6. Preview the learner experience, save the case, and export it for expert review. Retain case identifiers when analyzing items that share a vignette.

Each draft has five working tabs:

| Tab | Use it to |
| --- | --- |
| Write | Edit the vignette, blueprint, stages, items, response format, scoring, and every option with its rationale, type, proposed reasoning error, and selection role. Stage tools add, duplicate, remove, and reorder stages; 3- and 4-stage presets are editable scaffolds. A scoring preview shows how representative response patterns are scored. |
| Case map | See stages with their nested items and open findings, the evidence-to-decision map for each item, the blueprint trace, and reasoning-phase coverage. |
| Cognitive model | Record facts and when they are disclosed, feature values, constraint rules, and sources. A constraint matrix lists which authored combinations are permitted. Assemble variants, read why each option was included or excluded, and add the variants to the library. |
| Validate & review | Read structural errors, missing input, and expert-review prompts separately, then record named review decisions. Prompts include repeated options across items, later wording that reveals an earlier key, duplicate questions in one stage, and unmatched `{{placeholders}}`. |
| Learner preview | Answer the items as a learner would. Unfolding cases reveal one stage at a time and lock earlier answers. |

The Library dashboard above the tabs lists every draft with its type, review state, and open findings. **Archive** removes a draft from the working library; restore it from the dashboard. The sidebar imports a single case, a downloaded variant set, or a library backup, and nothing is imported if any case in the file is invalid.

Undo, redo, local commands, applied AI proposals, and dictation rebuild the form from the saved case, so open sections collapse to their defaults after those actions. The selected item and tab are kept.

Multiple-selection items require an explicit scoring policy and complete keyed set. Review option-level response patterns as well as the total; a total alone can conceal an important safety-related error.

## Manual mode and optional AI

Manual authoring does not need an AI model, API key, or network connection. Static reasoning, appraisal, and misconception guides remain available with AI assistance off. AI and voice requests occur only after the writer enables assistance and explicitly requests an operation. Turning AI off does not erase accepted content or its provenance.

Optional assistance is configured on the server through environment variables. Do not put credentials into the app's draft fields, exported cases, source files, or committed configuration. The app does not probe an AI service automatically.

| Variable | Purpose |
| --- | --- |
| `ITEMDEV_AI_PROVIDER` | `ollama` or `compatible` |
| `ITEMDEV_AI_BASE_URL` | Ollama service base URL, or a chat-completions-compatible server root/`/v1` base |
| `ITEMDEV_AI_MODEL` | Model name made available by the configured server |
| `ITEMDEV_AI_API_KEY` | Optional server credential, when the service requires one |
| `ITEMDEV_TRANSCRIBE_URL` | Full explicitly configured audio transcription endpoint |
| `ITEMDEV_TRANSCRIBE_MODEL` | Transcription model supplied to that endpoint |
| `ITEMDEV_TRANSCRIBE_API_KEY` | Optional transcription-service credential |

Only Ollama has a default base URL: `http://127.0.0.1:11434`. A model must still be configured. A compatible chat endpoint accepts a full `/chat/completions` URL, appends `/chat/completions` to a `/v1` base, or adds `/v1/chat/completions` when given only a server root. Use a server whose contract matches that interface.

Voice input, when configured, records a message and sends the WAV recording to the selected transcription endpoint. The adapter accepts recordings up to 10 MB and 180 seconds and uses a bounded timeout. This is recorded-message transcription, not a system-wide wake word or a verified live, hands-free realtime voice session. When no transcription endpoint/model is configured, voice is unavailable and manual editing continues. Existing operating-system dictation can still enter text independently.

AI suggestions remain drafts. The writer must inspect changes, source claims, and answer-key implications before adopting them. No live service connectivity or model quality is established merely by displaying a configured state.

## Clinical reasoning and appraisal guides

The proposed five-phase reasoning guide is **Identify & Interpret → Form & Prioritize → Decide → Act → Reassess & Update**. It is an editable teaching and authoring synthesis from the project discussion. It does not require five items or five stages, and it is not a validated five-factor measurement model.

The six proposed family medicine appraisal dimensions are:

- Clinical safety and triage.
- Patient-specific relevance.
- Clinical evidence and interpretation.
- Diagnostic reasoning and prioritization.
- Diagnostic testing appropriateness.
- Management and referral appropriateness.

These dimensions are a local authoring adaptation informed by [IMPACT](https://pmc.ncbi.nlm.nih.gov/articles/PMC13195341/) and [AACOM RACCCA](https://www.aacom.org/medical-education/ai-hub/ai-foundations/evaluating-ai-output). IMPACT was developed for evaluating generative AI in critical care; its published validation does not validate this family medicine adaptation or its learner scores. RACCCA's communication-quality criteria likewise do not replace clinical review.

Misconception labels are initially author hypotheses unless supported by expert review or actual learner evidence. Selecting a distractor does not independently establish what a learner thought or which cognitive bias occurred.

## Review and validity

Keep three forms of evidence separate:

1. **Structural checks:** Does the content satisfy the implemented schema, key, scoring, and disclosure rules?
2. **Clinical and item review:** Have appropriate human experts reviewed content, logic, wording, option plausibility, and the complete case sequence?
3. **Empirical evidence:** How do real learners respond, and do the resulting scores support the intended interpretation and use?

The intended workflow includes model-author review, independent model review, item/case review, pilot preparation, and a recorded release decision. AI cannot supply an independent human approval. Changing clinically meaningful content requires reassessment of affected review decisions.

Educational examples and generated variants are drafts. The app does not establish clinical correctness, operational exam validity, response independence, or certification readiness. It contains no real learner performance statistics unless actual response data have been explicitly supplied and analyzed. Related items from one vignette should retain their grouping identifiers for subsequent analysis.

## Project files

| File | Role |
| --- | --- |
| `app.py` | Streamlit entry point: library, dashboard, import/export, undo/redo, mode tabs |
| `ui_editor.py` | Manual authoring tabs: write, case map, cognitive model, review, learner preview |
| `ui_assistant.py` | Guidance panel, local typed commands, and optional AI proposals |
| `storage.py` | Atomic local library, revision history, archive, and session-only hosted storage |
| `cloud_entry.py` | Community Cloud entrypoint with allowlisted settings and session-only draft storage |
| `core.py` | Clinical case/item structures, constraints, scoring, validation, and export logic |
| `ai_adapter.py` | Optional guarded AI and recorded-voice transcription requests |
| `MASTER_PROMPT.md` | Complete product brief and acceptance criteria |
| `run.command` | Local launcher |
| `.streamlit/config.toml` | Local address, port, and theme |
| `test_core.py`, `test_ai_adapter.py`, `test_app.py`, `test_cloud_entry.py`, `test_ui_editor.py`, `test_ui_assistant.py`, `test_workflows.py` | Domain, adapter, persistence, hosted-entry, and simulated interface checks |

Local runtime data and secrets are excluded by `.gitignore`. Use the app's export controls to retain portable copies of authored material. Review [MASTER_PROMPT.md](MASTER_PROMPT.md) for the distinction between the original Shiny example, published methods, and proposed extensions.

Local `app.py` saves its library to `data/library.json` using atomic replacement. Field changes are saved when committed by the control, such as after leaving a text field, and explicit editing operations save immediately. Undo/redo history is persisted per case and is bounded to 30 entries. Review records are associated with the content version; a saved review is not a clinical-validity certificate. Back up the exported library as well as the local data directory when preserving a project. Hosted `cloud_entry.py` uses session-only storage instead.

For an isolated local test or alternate local data folder, set `ITEMDEV_DATA_DIR` to a directory of your choice. Run the focused checks from this folder with the same Python environment as the app:

```bash
../.venv/bin/python -m unittest test_core test_ai_adapter test_app test_cloud_entry test_ui_editor test_ui_assistant test_workflows
```

The interface checks use Streamlit AppTest, a simulated runtime. It cannot show whether a browser field refreshes after the saved case changes, so undo, local commands, applied proposals, item counts, table saves, import, variant assembly, and archive/restore were also exercised once in desktop Chrome against a stand-in AI server. Neither kind of check establishes live microphone behavior, clinical correctness, or live AI service performance.

Not yet built from [MASTER_PROMPT.md](MASTER_PROMPT.md): hands-free or streaming voice (recorded messages only), import and analysis of learner response data, response-dependent branching, a graphical model diagram, an editable framework crosswalk, and conversational commands beyond the exact local vocabulary.

No hosting, publication, Git commit, or upload is performed by the launcher.

The deployment entry point uses a space-free path because Community Cloud dependency installation does not quote requirement paths containing spaces. The working app remains in `item dev`; the wrapper loads its hosted entry point.

## Family Medicine decision workflow

Open **Cognitive model → Family Medicine decision workflow** for the selected item. Set the decision, clinical setting, population, intended examinees, assessment intent, and any proposed key. Link decision features to fact IDs in **Shared facts and disclosure**. Keep the values that define the clinical scenario separate from which observations the learner has seen.

1. Enter approved bank summaries and clinical passages with their source class, date or version, and locator. Use the `providers` list to grant OpenAI or Anthropic access to each record. A record without a provider stays available for local review. Scope a record to an item, stage, population, or setting as appropriate.
2. Review the clinical claims and decision model. Use the local scenario table to inspect the derived key, feasibility, missing information, and triggered rules. Confirm the selected scenario and action key before drafting. A fact or disclosure edit refreshes the local evaluation and the review status.
3. Review the wording and options for that scenario. The option pool shows why each alternative is attractive, the rule it fails, and any shortfall. Link manually authored modeled options to their actions and record the approved option IDs after assembly.
4. Record a blinded solve before comparing it with the approved key. Run cueing separately for unfolding sequences or record a human cueing decision. Record the human clinical decision and resolve material disagreements before pilot or release.
5. Review rationale proposals and retain the full JSON or author Markdown with the scenario, sources, option IDs, and revision history. A linked variant includes the source case, item, and scenario IDs.

The optional **Family Medicine specialists** panel shows the effective route and the projected request. Each button invocation sends one request. The versioned registry and its implementation specification are in [item dev 2](../item%20dev%202/). Each run records the registry version. After editing the registry's prompts, schemas, routing or request projection, bump its `version` and record the new fingerprint in `test_registry_version.py`. General AI assistance and recorded transcription retain their own settings.

| Local environment or hosted Streamlit Secrets | Purpose |
| --- | --- |
| `OPENAI_API_KEY` and `ITEMDEV_OPENAI_MODEL` | Default credential and model for OpenAI Responses |
| `ANTHROPIC_API_KEY` and `ITEMDEV_ANTHROPIC_MODEL` | Default credential and model for Anthropic Messages |
| `ITEMDEV_AGENT_ROUTES` | Optional JSON mapping of role IDs to a provider and optional model override |

For example, `{"rationale_writing":{"provider":"anthropic"}}` uses the Anthropic model setting for that role. Deployments need approved model IDs and server-side credentials for the roles they intend to use. Limit authoring access to trusted writers and reviewers through the hosting environment. The app records the entered reviewer identity and decision at the content revision.

### Feature space and distractor space

After accepting a decision model, choose **Scenario to review**. **Feature space** shows the permitted values, linked facts, disclosure stage, rules and sources for each feature. Inspect coupled features or try a change in a visible variable to see the evaluated key before saving. **Distractor space** uses the same scenario. It shows nearby scenarios where each action is keyed, its exclusion rules, and the number of distinct eligible options. A reviewer can retain a candidate, select it for assembly, or revise its judgment. Confirming a changed scenario or editing a fact prompts reassessment of prior judgments.

### Search an approved collection

In **Search bank and clinical references**, load a structured bank as a JSON array. For clinical references, choose an approved Markdown folder locally or upload Markdown or JSON files. The local connector reads the selected folder recursively. It skips hidden entries and symbolic links and leaves the original files untouched. Uploaded files and the search index stay in the session. Only the excerpts selected by a named writer are saved with the item and stage.

A minimal synthetic bank import looks like this.

```json
[{
  "source_id": "demo-1", "source_class": "bank", "version": "2",
  "locator": "Demo bank / item 1", "summary": "Select a next action once a report is available.",
  "decision_timing": "Report review", "option_type": "Management",
  "blueprint": {"domain": "Synthetic exercise"}, "options": [{"text": "Action A"}],
  "providers": []
}]
```

Markdown headings identify the section for each passage. Add properties at the top of the note when you know the original source, date or version, and applicable scope.

```markdown
---
source_class: evidence summary
version: Fixture v1
citation: Synthetic report, section 2
decision_timing: Report review
providers: []
---
# When the report arrives
The synthetic exercise distinguishes a final result from a pending report.
```

Use `approved: false` to exclude a record or note. Explicit record-level provider lists narrow the collection grant. A Markdown note without a provider property inherits the grant set by the writer when loading it. A structured JSON record without a provider list stays local. Both formats default to local-only when the collection has no provider grant. Replacing a collection rechecks saved permissions. Writers can narrow a saved grant or remove a passage from the item.

Search runs locally using concept synonyms, TF-IDF similarity and, for larger collections, a low-rank representation of co-occurring terms. The query uses the item's objective, decision, reasoning demand, population, setting, and active disclosure. Exact terms can prioritize a named test or threshold. Each match shows its passage, locator, scope and reason for matching. A similarity score describes retrieval relevance. Observed option performance remains tied to its bank record, version, population and administration.

Practice notes can suggest context. A retrieved guideline, study or evidence summary needs an original citation to enter clinical evidence review. The SME must mark the extracted claim applicable before linking it to a decision rule. The authoring views retain source references for reviewed features and candidates. The **Bank summaries and evidence passages** editor also accepts manual excerpts when a collection is unavailable.

Run the Item Studio suite from the repository root with `.venv/bin/python -m pytest -q 'item dev'`. The local and hosted entrypoints use synthetic fixtures and mock provider transports in the tests.
