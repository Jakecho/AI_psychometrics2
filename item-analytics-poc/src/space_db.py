"""Dedicated SQLite database for reviewed feature spaces (key-feature, instance and distractor spaces).

Only SME-reviewed spaces are stored here, by explicit save. Item banks and
embedding vectors are never stored. Every write bumps the space's version;
callers pass the version they read so concurrent edits are refused instead of
overwritten. Removals are soft (restorable); only an explicit purge of a
deleted space removes rows, and the history keeps a tombstone.

The file is shared by everyone who can reach it. It is enabled by default only
for local runs; hosted deployments must opt in with ``ITEM_ANALYTICS_DB_PATH``.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid

import pandas as pd

from .schemas import DOMINANCE, FEATURE_TYPE_LABELS, FEATURE_TYPES, REVIEW_STATUSES

DB_SCHEMA_VERSION = 1
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "space_library.sqlite"
ENV_VAR = "ITEM_ANALYTICS_DB_PATH"
ELEMENT_EDITABLE = ("label", "definition", "status", "importance", "dominance", "procedure_group", "sequence_order",
                    "note", "misconception")
SPACE_EDITABLE = ("task_name", "decision_or_action", "domain", "notes", "workspace")
EDITABLE_STATUSES = tuple(s for s in REVIEW_STATUSES if s not in ("merged", "removed"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spaces (
    space_id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    task_name TEXT NOT NULL,
    decision_or_action TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    target_item_id TEXT,
    target_preview TEXT,
    sibling_ids TEXT NOT NULL DEFAULT '[]',
    source_name TEXT,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, created_by TEXT NOT NULL,
    updated_at TEXT NOT NULL, updated_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS elements (
    element_id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(space_id) ON DELETE CASCADE,
    feature_id TEXT NOT NULL,
    space_type TEXT NOT NULL,
    label TEXT NOT NULL,
    definition TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    removed_from_status TEXT,
    importance INTEGER CHECK (importance IS NULL OR importance BETWEEN 1 AND 5),
    dominance TEXT, procedure_group TEXT, sequence_order INTEGER,
    supporting_items TEXT NOT NULL DEFAULT '[]',
    option_refs TEXT NOT NULL DEFAULT '[]',
    evidence_types TEXT NOT NULL DEFAULT '[]',
    source_spans TEXT NOT NULL DEFAULT '[]',
    inference_rationale TEXT NOT NULL DEFAULT '',
    misconception TEXT,
    note TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'ai',
    updated_at TEXT NOT NULL, updated_by TEXT NOT NULL,
    UNIQUE (space_id, feature_id)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    space_id TEXT NOT NULL, element_id TEXT, action TEXT NOT NULL,
    before TEXT, after TEXT, actor TEXT NOT NULL, at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_elements_space ON elements(space_id);
CREATE INDEX IF NOT EXISTS ix_spaces_lookup ON spaces(workspace, task_name, domain);
CREATE INDEX IF NOT EXISTS ix_history_space ON history(space_id);
"""


class SpaceDBError(ValueError):
    pass


class ConflictError(SpaceDBError):
    """The space changed since the caller read it."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(env=None, secrets=None, hosted=None):
    """Return (path or None, reason). Hosted deployments are off unless explicitly configured."""
    env = os.environ if env is None else env
    value = (env.get(ENV_VAR) or (secrets or {}).get(ENV_VAR) or "").strip()
    if value.lower() in ("off", "none", "disabled", "0", "false"):
        return None, "The space database is disabled by configuration."
    if value:
        return Path(value).expanduser(), f"Configured database: {Path(value).name}"
    if hosted is None:
        hosted = Path("/mount/src").exists()  # Streamlit Community Cloud checkout location
    if hosted:
        return None, (f"The space database is off on hosted deployments. An administrator can set {ENV_VAR}; "
                      "the library is then shared by everyone with access to the app.")
    return DEFAULT_PATH, "Local database (data/space_library.sqlite)"


def _j(value):
    return json.dumps(value if value is not None else [], ensure_ascii=False, default=str)


def _u(text, default=None):
    if text in (None, ""):
        return [] if default is None else default
    return json.loads(text)


class SpaceDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(DB_SCHEMA_VERSION),))
            elif int(row["value"]) != DB_SCHEMA_VERSION:
                raise SpaceDBError(f"Unsupported space database schema version {row['value']}.")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _log(conn, space_id, element_id, action, before, after, actor):
        conn.execute("INSERT INTO history (space_id, element_id, action, before, after, actor, at) VALUES (?,?,?,?,?,?,?)",
                     (space_id, element_id, action, None if before is None else _j(before),
                      None if after is None else _j(after), actor or "anonymous", now()))

    @staticmethod
    def _space_row(conn, space_id):
        row = conn.execute("SELECT * FROM spaces WHERE space_id = ?", (space_id,)).fetchone()
        if row is None:
            raise SpaceDBError(f"Space {space_id} is not in the database.")
        return row

    def _check_version(self, conn, space_id, expected_version):
        row = self._space_row(conn, space_id)
        if expected_version is not None and row["version"] != expected_version:
            raise ConflictError(f"Space {space_id} changed since you opened it (version {expected_version} → "
                                f"{row['version']}). Reload it before saving, or save your version as a new space.")
        return row

    @staticmethod
    def _bump(conn, space_id, actor):
        conn.execute("UPDATE spaces SET version = version + 1, updated_at = ?, updated_by = ? WHERE space_id = ?",
                     (now(), actor or "anonymous", space_id))
        return conn.execute("SELECT version FROM spaces WHERE space_id = ?", (space_id,)).fetchone()["version"]

    @staticmethod
    def _element_values(f):
        return {
            "feature_id": f["feature_id"], "space_type": f["feature_type"], "label": f["canonical_label"],
            "definition": f.get("definition") or "", "status": f["review_status"],
            "removed_from_status": f.get("removed_from_status"), "importance": f.get("sme_importance"),
            "dominance": f.get("dominance"), "procedure_group": f.get("procedure_group"),
            "sequence_order": f.get("sequence_order"), "supporting_items": _j(f.get("supporting_items")),
            "option_refs": _j(f.get("option_refs")), "evidence_types": _j(f.get("evidence_types")),
            "source_spans": _j(f.get("source_spans")), "inference_rationale": f.get("inference_rationale") or "",
            "misconception": f.get("proposed_misconception"), "note": f.get("sme_note") or "",
            "origin": "sme" if f.get("origin") == "sme" else "ai",
        }

    # ------------------------------------------------------------------ save from a session

    def save_space(self, workspace, task_review, store, target, sibling_ids, actor, source_name=None,
                   space_id=None, expected_version=None):
        """Insert a new space, or update ``space_id`` when ``expected_version`` matches. Returns (space_id, version)."""
        if not task_review or task_review.get("task_status") not in ("accepted", "edited") or not (task_review.get("task") or {}).get("name"):
            raise SpaceDBError("Confirm the target task before saving a space; spaces are filed by task.")
        if not (workspace or "").strip():
            raise SpaceDBError("Enter a workspace (library) name.")
        task = task_review["task"]
        features = [store["features"][fid] for fid in store["order"] if store["features"][fid]["review_status"] != "merged"]
        if not features:
            raise SpaceDBError("The space has no elements to save.")
        stamp, actor = now(), actor or "anonymous"
        preview = " ".join(v for v in target.get("fields", {}).values() if v)[:240]
        meta = {"workspace": workspace.strip(), "task_name": task["name"], "decision_or_action": task.get("decision_or_action") or "",
                "domain": task.get("domain") or "", "target_item_id": target["item_id"], "target_preview": preview,
                "sibling_ids": _j(list(sibling_ids)), "source_name": source_name}
        with self.connect() as conn:
            if space_id is None:
                space_id = uuid.uuid4().hex[:12]
                conn.execute(
                    "INSERT INTO spaces (space_id, workspace, task_name, decision_or_action, domain, target_item_id, "
                    "target_preview, sibling_ids, source_name, created_at, created_by, updated_at, updated_by) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (space_id, meta["workspace"], meta["task_name"], meta["decision_or_action"], meta["domain"],
                     meta["target_item_id"], meta["target_preview"], meta["sibling_ids"], meta["source_name"],
                     stamp, actor, stamp, actor))
                self._log(conn, space_id, None, "create_space", None, {"task": meta["task_name"], "elements": len(features)}, actor)
                existing = {}
            else:
                self._check_version(conn, space_id, expected_version)
                conn.execute("UPDATE spaces SET workspace=?, task_name=?, decision_or_action=?, domain=?, target_item_id=?, "
                             "target_preview=?, sibling_ids=?, source_name=?, status='active' WHERE space_id=?",
                             (meta["workspace"], meta["task_name"], meta["decision_or_action"], meta["domain"],
                              meta["target_item_id"], meta["target_preview"], meta["sibling_ids"], meta["source_name"], space_id))
                existing = {r["feature_id"]: dict(r) for r in conn.execute("SELECT * FROM elements WHERE space_id=?", (space_id,))}
            seen = set()
            for f in features:
                values = self._element_values(f)
                seen.add(values["feature_id"])
                eid = f"{space_id}:{values['feature_id']}"
                old = existing.get(values["feature_id"])
                if old is None:
                    cols = ["element_id", "space_id", *values, "updated_at", "updated_by"]
                    conn.execute(f"INSERT INTO elements ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                 [eid, space_id, *values.values(), stamp, actor])
                    if space_id and existing:
                        self._log(conn, space_id, eid, "add_element", None, {"label": values["label"]}, actor)
                else:
                    changed = {k: v for k, v in values.items() if old.get(k) != v}
                    if changed:
                        conn.execute(f"UPDATE elements SET {', '.join(f'{k}=?' for k in changed)}, updated_at=?, updated_by=? "
                                     "WHERE element_id=?", [*changed.values(), stamp, actor, eid])
                        self._log(conn, space_id, eid, "update_element", {k: old[k] for k in changed}, changed, actor)
            for fid, old in existing.items():
                if fid not in seen and old["status"] != "removed":
                    conn.execute("UPDATE elements SET removed_from_status=status, status='removed', updated_at=?, updated_by=? "
                                 "WHERE element_id=?", (stamp, actor, old["element_id"]))
                    self._log(conn, space_id, old["element_id"], "removed_on_save", {"status": old["status"]}, {"status": "removed"}, actor)
            if existing or expected_version is not None:
                self._log(conn, space_id, None, "save_space", None, {"elements": len(features)}, actor)
                version = self._bump(conn, space_id, actor)
            else:
                version = 1
        return space_id, version

    # ------------------------------------------------------------------ read and search

    def spaces(self, workspace=None, tasks=None, domains=None, text="", include_deleted=False):
        sql = ["SELECT s.*, "
               + ", ".join(f"SUM(CASE WHEN e.space_type='{t}' AND e.status!='removed' THEN 1 ELSE 0 END) AS \"{FEATURE_TYPE_LABELS[t]}\""
                           for t in FEATURE_TYPES)
               + ", SUM(CASE WHEN e.status='removed' THEN 1 ELSE 0 END) AS removed "
               "FROM spaces s LEFT JOIN elements e ON e.space_id = s.space_id WHERE 1=1"]
        params = []
        self._filters(sql, params, workspace, tasks, domains, include_deleted, prefix="s.")
        if text:
            sql.append("AND (s.task_name LIKE ? ESCAPE '\\' OR s.domain LIKE ? ESCAPE '\\' OR s.notes LIKE ? ESCAPE '\\')")
            params += [_like(text)] * 3
        sql.append("GROUP BY s.space_id ORDER BY s.updated_at DESC, s.space_id")
        with self.connect() as conn:
            rows = [dict(r) for r in conn.execute(" ".join(sql), params)]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["space_id", "workspace", "task_name", "domain", "status", "version"])
        frame["siblings_N"] = frame["sibling_ids"].map(lambda s: len(_u(s)))
        cols = ["space_id", "workspace", "task_name", "decision_or_action", "domain", "target_item_id", "siblings_N",
                *[FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES], "removed", "status", "version", "notes",
                "source_name", "updated_at", "updated_by"]
        for c in [FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES] + ["removed"]:
            frame[c] = frame[c].fillna(0).astype(int)
        return frame[cols]

    @staticmethod
    def _filters(sql, params, workspace, tasks, domains, include_deleted, prefix):
        if workspace:
            sql.append(f"AND {prefix}workspace = ?")
            params.append(workspace)
        if tasks:
            sql.append(f"AND {prefix}task_name IN ({', '.join('?' * len(tasks))})")
            params += list(tasks)
        if domains:
            sql.append(f"AND {prefix}domain IN ({', '.join('?' * len(domains))})")
            params += list(domains)
        if not include_deleted:
            sql.append(f"AND {prefix}status = 'active'")

    def elements(self, workspace=None, tasks=None, domains=None, space_types=None, statuses=None, text="",
                 include_removed=False, include_deleted=False, space_id=None):
        sql = ["SELECT e.*, s.workspace, s.task_name, s.domain, s.target_item_id, s.sibling_ids, s.version AS space_version, "
               "s.status AS space_status FROM elements e JOIN spaces s ON s.space_id = e.space_id WHERE 1=1"]
        params = []
        self._filters(sql, params, workspace, tasks, domains, include_deleted, prefix="s.")
        if space_id:
            sql.append("AND e.space_id = ?")
            params.append(space_id)
        if space_types:
            sql.append(f"AND e.space_type IN ({', '.join('?' * len(space_types))})")
            params += list(space_types)
        if statuses:
            sql.append(f"AND e.status IN ({', '.join('?' * len(statuses))})")
            params += list(statuses)
        if not include_removed:
            sql.append("AND e.status != 'removed'")
        if text:
            sql.append("AND (e.label LIKE ? ESCAPE '\\' OR e.definition LIKE ? ESCAPE '\\' OR e.note LIKE ? ESCAPE '\\' "
                       "OR e.misconception LIKE ? ESCAPE '\\' OR s.task_name LIKE ? ESCAPE '\\' OR s.domain LIKE ? ESCAPE '\\')")
            params += [_like(text)] * 6
        sql.append("ORDER BY s.task_name, s.space_id, e.space_type, e.feature_id")
        with self.connect() as conn:
            rows = [dict(r) for r in conn.execute(" ".join(sql), params)]
        out = []
        for r in rows:
            sibs = set(_u(r["sibling_ids"]))
            support = _u(r["supporting_items"])
            n = len(sibs)
            hits = len([i for i in dict.fromkeys(support) if i in sibs])
            out.append({
                "element_id": r["element_id"], "space_id": r["space_id"], "workspace": r["workspace"],
                "task": r["task_name"], "domain": r["domain"], "space_type": FEATURE_TYPE_LABELS.get(r["space_type"], r["space_type"]),
                "feature_id": r["feature_id"], "label": r["label"], "definition": r["definition"], "status": r["status"],
                "importance": r["importance"], "dominance": r["dominance"], "procedure_group": r["procedure_group"],
                "sequence_order": r["sequence_order"], "note": r["note"], "misconception": r["misconception"],
                "origin": "SME-added" if r["origin"] == "sme" else "AI proposal",
                "coverage": f"{hits}/{n} = {100 * hits / n:.0f}%" if n else "N/A",
                "supporting_items": ", ".join(support),
                "options": ", ".join(f"{o['item_id']}:{o['option_id']}" for o in _u(r["option_refs"])),
                "evidence": ", ".join(_u(r["evidence_types"])), "space_version": r["space_version"],
                "updated_at": r["updated_at"], "updated_by": r["updated_by"],
            })
        return pd.DataFrame(out, columns=["element_id", "space_id", "workspace", "task", "domain", "space_type", "feature_id",
                                          "label", "definition", "status", "importance", "dominance", "procedure_group",
                                          "sequence_order", "note", "misconception", "origin", "coverage",
                                          "supporting_items", "options", "evidence", "space_version", "updated_at", "updated_by"])

    def get_space(self, space_id):
        with self.connect() as conn:
            space = dict(self._space_row(conn, space_id))
            elements = [dict(r) for r in conn.execute("SELECT * FROM elements WHERE space_id=? ORDER BY feature_id", (space_id,))]
        space["sibling_ids"] = _u(space["sibling_ids"])
        for e in elements:
            for k in ("supporting_items", "option_refs", "evidence_types", "source_spans"):
                e[k] = _u(e[k])
        return {"space": space, "elements": elements}

    def history(self, space_id=None, limit=200):
        sql, params = "SELECT * FROM history", []
        if space_id:
            sql += " WHERE space_id = ?"
            params.append(space_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params)]
        return pd.DataFrame(rows, columns=["id", "space_id", "element_id", "action", "before", "after", "actor", "at"])

    def facets(self, workspace=None, include_deleted=False):
        sql, params = ["SELECT DISTINCT workspace, task_name, domain FROM spaces WHERE 1=1"], []
        self._filters(sql, params, workspace, None, None, include_deleted, prefix="")
        with self.connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        with self.connect() as conn:
            workspaces = sorted({r["workspace"] for r in conn.execute("SELECT DISTINCT workspace FROM spaces")})
        return {"workspaces": workspaces, "tasks": sorted({r["task_name"] for r in rows}),
                "domains": sorted({r["domain"] for r in rows})}

    # ------------------------------------------------------------------ updates

    def update_element(self, element_id, changes, actor, expected_version):
        clean = {}
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM elements WHERE element_id=?", (element_id,)).fetchone()
            if row is None:
                raise SpaceDBError(f"Element {element_id} is not in the database.")
            self._check_version(conn, row["space_id"], expected_version)
            for key, value in changes.items():
                if key not in ELEMENT_EDITABLE:
                    raise SpaceDBError(f"'{key}' cannot be edited here.")
                if key == "label" and not (value or "").strip():
                    raise SpaceDBError("A label cannot be empty.")
                if key == "status" and value not in EDITABLE_STATUSES:
                    raise SpaceDBError(f"Status must be one of {', '.join(EDITABLE_STATUSES)}; use Remove/Restore for removal.")
                if key == "importance" and value not in (None, 1, 2, 3, 4, 5):
                    raise SpaceDBError("Importance must be unrated or 1–5.")
                if key in ("dominance", "procedure_group", "sequence_order") and row["space_type"] != "procedural_key" and value not in (None, ""):
                    raise SpaceDBError("Dominance, group and order apply to procedural key features only.")
                if key == "dominance" and value not in (None, "") and value not in DOMINANCE:
                    raise SpaceDBError(f"Dominance must be one of {', '.join(DOMINANCE)}.")
                if key == "sequence_order" and value not in (None, ""):
                    value = int(value)
                    if value < 1:
                        raise SpaceDBError("Order starts at 1.")
                if value == "":
                    value = None if key in ("dominance", "procedure_group", "sequence_order", "misconception") else value
                if row[key] != value:
                    clean[key] = value
            if row["status"] == "removed" and "status" in clean:
                raise SpaceDBError("Restore the element before changing its status.")
            if not clean:
                return self._space_row(conn, row["space_id"])["version"]
            conn.execute(f"UPDATE elements SET {', '.join(f'{k}=?' for k in clean)}, updated_at=?, updated_by=? WHERE element_id=?",
                         [*clean.values(), now(), actor or "anonymous", element_id])
            self._log(conn, row["space_id"], element_id, "update_element", {k: row[k] for k in clean}, clean, actor)
            return self._bump(conn, row["space_id"], actor)

    def set_removed(self, element_ids, removed, actor, expected_versions, reason=""):
        """Remove (soft) or restore elements. ``expected_versions`` maps space_id → version read."""
        done = []
        with self.connect() as conn:
            touched = set()
            for eid in element_ids:
                row = conn.execute("SELECT * FROM elements WHERE element_id=?", (eid,)).fetchone()
                if row is None:
                    raise SpaceDBError(f"Element {eid} is not in the database.")
                if row["space_id"] not in touched:
                    self._check_version(conn, row["space_id"], expected_versions.get(row["space_id"]))
                    touched.add(row["space_id"])
                if removed and row["status"] != "removed":
                    conn.execute("UPDATE elements SET removed_from_status=status, status='removed', updated_at=?, updated_by=? "
                                 "WHERE element_id=?", (now(), actor or "anonymous", eid))
                    self._log(conn, row["space_id"], eid, "remove_element", {"status": row["status"]}, {"status": "removed", "reason": reason}, actor)
                    done.append(eid)
                elif not removed and row["status"] == "removed":
                    restored = row["removed_from_status"] or "pending"
                    conn.execute("UPDATE elements SET status=?, removed_from_status=NULL, updated_at=?, updated_by=? WHERE element_id=?",
                                 (restored, now(), actor or "anonymous", eid))
                    self._log(conn, row["space_id"], eid, "restore_element", {"status": "removed"}, {"status": restored}, actor)
                    done.append(eid)
            for sid in touched:
                self._bump(conn, sid, actor)
        return done

    def update_space(self, space_id, changes, actor, expected_version):
        with self.connect() as conn:
            row = self._check_version(conn, space_id, expected_version)
            clean = {}
            for key, value in changes.items():
                if key not in SPACE_EDITABLE:
                    raise SpaceDBError(f"'{key}' cannot be edited here.")
                if key in ("task_name", "workspace") and not (value or "").strip():
                    raise SpaceDBError(f"{key.replace('_', ' ').capitalize()} cannot be empty.")
                if row[key] != value:
                    clean[key] = value
            if not clean:
                return row["version"]
            conn.execute(f"UPDATE spaces SET {', '.join(f'{k}=?' for k in clean)} WHERE space_id=?", [*clean.values(), space_id])
            self._log(conn, space_id, None, "update_space", {k: row[k] for k in clean}, clean, actor)
            return self._bump(conn, space_id, actor)

    def set_space_deleted(self, space_id, deleted, actor, expected_version):
        with self.connect() as conn:
            row = self._check_version(conn, space_id, expected_version)
            status = "deleted" if deleted else "active"
            if row["status"] == status:
                return row["version"]
            conn.execute("UPDATE spaces SET status=? WHERE space_id=?", (status, space_id))
            self._log(conn, space_id, None, "delete_space" if deleted else "restore_space",
                      {"status": row["status"]}, {"status": status}, actor)
            return self._bump(conn, space_id, actor)

    def purge_space(self, space_id, actor, confirm=False):
        """Permanently delete a space that is already deleted. The history keeps a tombstone."""
        if not confirm:
            raise SpaceDBError("Confirm the permanent purge.")
        with self.connect() as conn:
            row = self._space_row(conn, space_id)
            if row["status"] != "deleted":
                raise SpaceDBError("Delete the space first; only deleted spaces can be purged.")
            labels = [r["label"] for r in conn.execute("SELECT label FROM elements WHERE space_id=?", (space_id,))]
            conn.execute("DELETE FROM spaces WHERE space_id=?", (space_id,))
            self._log(conn, space_id, None, "purge_space", {"task": row["task_name"], "domain": row["domain"],
                                                             "elements": len(labels)}, None, actor)

    # ------------------------------------------------------------------ backup

    def export_json(self, workspace=None, include_deleted=True):
        with self.connect() as conn:
            sql, params = "SELECT * FROM spaces", []
            if workspace:
                sql += " WHERE workspace = ?"
                params.append(workspace)
            spaces = [dict(r) for r in conn.execute(sql, params)]
            if not include_deleted:
                spaces = [s for s in spaces if s["status"] == "active"]
            for s in spaces:
                s["elements"] = [dict(r) for r in conn.execute("SELECT * FROM elements WHERE space_id=?", (s["space_id"],))]
        return {"format": "item-analytics-space-library", "db_schema_version": DB_SCHEMA_VERSION, "exported_at": now(),
                "spaces": spaces}

    def import_json(self, data, actor, on_conflict="skip"):
        """Restore spaces from ``export_json`` output. Existing space IDs are skipped or replaced."""
        if not isinstance(data, dict) or data.get("format") != "item-analytics-space-library":
            raise SpaceDBError("This is not a space library backup.")
        if data.get("db_schema_version") != DB_SCHEMA_VERSION:
            raise SpaceDBError("The backup was made with an incompatible database version.")
        if on_conflict not in ("skip", "replace"):
            raise SpaceDBError("on_conflict must be 'skip' or 'replace'.")
        counts = {"added": 0, "replaced": 0, "skipped": 0}
        space_cols = [c for c in ("space_id", "workspace", "task_name", "decision_or_action", "domain", "target_item_id",
                                  "target_preview", "sibling_ids", "source_name", "notes", "status", "version", "created_at",
                                  "created_by", "updated_at", "updated_by")]
        with self.connect() as conn:
            element_cols = [r["name"] for r in conn.execute("PRAGMA table_info(elements)")]
            for s in data.get("spaces", []):
                exists = conn.execute("SELECT 1 FROM spaces WHERE space_id=?", (s["space_id"],)).fetchone()
                if exists and on_conflict == "skip":
                    counts["skipped"] += 1
                    continue
                if exists:
                    conn.execute("DELETE FROM spaces WHERE space_id=?", (s["space_id"],))
                conn.execute(f"INSERT INTO spaces ({', '.join(space_cols)}) VALUES ({', '.join('?' * len(space_cols))})",
                             [s.get(c) for c in space_cols])
                for e in s.get("elements", []):
                    if e.get("space_type") not in FEATURE_TYPES:
                        raise SpaceDBError(f"Backup element {e.get('element_id')} has an unknown space type.")
                    conn.execute(f"INSERT INTO elements ({', '.join(element_cols)}) VALUES ({', '.join('?' * len(element_cols))})",
                                 [e.get(c) for c in element_cols])
                self._log(conn, s["space_id"], None, "import_space", None, {"replaced": bool(exists)}, actor)
                counts["replaced" if exists else "added"] += 1
        return counts


def _like(text):
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def editor_changes(before_rows, after_rows):
    """Translate edited database-table rows into ``update_element`` changes keyed by element ID (never by position)."""
    def norm(key, value):
        if value is None or (isinstance(value, float) and value != value) or value == "":
            return None
        if key == "importance":
            return None if value == "unrated" else int(value)
        if key == "sequence_order":
            return int(value)
        return value

    before = {r["element_id"]: r for r in before_rows}
    out = {}
    for after in after_rows:
        prev = before.get(after["element_id"])
        if prev is None:
            continue
        changes = {}
        for key in ELEMENT_EDITABLE:
            if key in after and norm(key, after[key]) != norm(key, prev[key]):
                changes[key] = norm(key, after[key])
        if changes.get("status") == "removed" or (prev.get("status") == "removed" and "status" in changes):
            changes.pop("status")  # removal and restore have their own actions
        if "label" in changes and changes["label"] is None:
            changes["label"] = ""
        if "definition" in changes and changes["definition"] is None:
            changes["definition"] = ""
        if "note" in changes and changes["note"] is None:
            changes["note"] = ""
        if changes:
            out[after["element_id"]] = (changes, int(prev["space_version"]))
    return out


def apply_editor_changes(db, changes, actor):
    """Apply ``editor_changes`` output. Edits to several elements of one space chain the version each write returns."""
    versions, errors, applied = {}, [], []
    for eid, (change, version) in changes.items():
        space_id = eid.split(":", 1)[0]
        try:
            versions[space_id] = db.update_element(eid, change, actor, versions.get(space_id, version))
            applied.append(eid)
        except SpaceDBError as exc:
            errors.append(f"{eid}: {exc}")
    return applied, errors
