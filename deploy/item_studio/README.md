# Item Studio deployment entrypoint

Configure Streamlit Community Cloud with:

| Setting | Value |
| --- | --- |
| Repository | `Jakecho/AI_psychometrics2` |
| Branch | `main` |
| Main file path | `deploy/item_studio/streamlit_app.py` |
| Python version | `3.13` |

This folder intentionally has no spaces in its path. Its `requirements.txt` is identical to the minimal requirements in `item dev`, so Community Cloud can install the application dependencies from a path that does not require quoting.

`streamlit_app.py` delegates to `item dev/cloud_entry.py`. Application development files remain in `item dev`; the wrapper does not duplicate or replace them. The hosted entrypoint supplies the supported Streamlit Secrets mapping and isolated, session-only draft storage. It does not activate a shared on-disk library for visitors.

Export drafts before refreshing the browser, disconnecting, restarting, or redeploying the hosted app. Hosted storage is not durable. AI remains optional and off by default; no credentials are needed for the manual authoring workflow. Configure optional AI and transcription settings in Streamlit Secrets. For the seven Family Medicine specialists, include the approved OpenAI and Anthropic credentials and model IDs for the routes you enable. Keep both `item dev` and `item dev 2` in the checkout so the hosted entrypoint can load the versioned registry. See the [Item Studio deployment and authoring notes](../../item%20dev/README.md).

For local development with durable JSON storage, continue using `item dev/run.command` or `item dev/app.py`.

The hosted **Search bank and clinical references** control accepts approved JSON bank records and clinical Markdown or JSON uploads. Search runs inside the session. Selected passages and their permissions are included in the exported case. Folder indexing is available through the local entrypoint.
