"""CSV/XLSX parsing, schema profiling, column mapping and canonical item records."""

from dataclasses import asdict, dataclass, field
import io
import math
import re

import pandas as pd

from .text_builder import build_embed_text, fingerprint, sha256_text, template_description

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ITEMS = 1000
MAX_COLUMNS = 300
MAX_CELL_CHARS = 50_000

ROLE_FIELDS = ("domain", "task", "category", "difficulty", "p_value", "point_biserial",
               "enemy", "rationale", "family", "scale_group")
NUMERIC_ROLES = ("difficulty", "p_value", "point_biserial")


class IngestError(ValueError):
    """An actionable problem with the uploaded file or mapping."""


@dataclass
class ColumnMapping:
    item_id: str | None = None
    text_columns: list = field(default_factory=list)
    text_labels: dict = field(default_factory=dict)
    option_columns: list = field(default_factory=list)
    option_ids: list | None = None
    key: str | None = None
    domain: str | None = None
    task: str | None = None
    category: str | None = None
    difficulty: str | None = None
    p_value: str | None = None
    point_biserial: str | None = None
    enemy: str | None = None
    rationale: str | None = None
    family: str | None = None
    scale_group: str | None = None
    include_options_in_embed: bool = False
    type_overrides: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def label(self, column):
        return self.text_labels.get(column) or column

    def embed_settings(self):
        """The parts of the mapping that change the embedded text."""
        return {
            "text_columns": list(self.text_columns),
            "labels": [self.label(c) for c in self.text_columns],
            "option_columns": list(self.option_columns) if self.include_options_in_embed else [],
            "include_options": bool(self.include_options_in_embed),
        }


# --------------------------------------------------------------------------- reading

def list_sheets(data, filename):
    if not filename.lower().endswith((".xlsx", ".xls")):
        return []
    try:
        return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)
    except Exception as exc:  # noqa: BLE001 - surface parser errors as user messages
        raise IngestError(f"The workbook could not be opened: {type(exc).__name__}.") from None


def read_table(data, filename, sheet=None):
    """Read CSV/XLSX bytes into a string-typed DataFrame without losing values."""
    if not data:
        raise IngestError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise IngestError(f"The file exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit for this proof of concept.")
    name = filename.lower()
    if name.endswith(".csv"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestError(
                f"The CSV is not valid UTF-8 (byte {exc.start}). Re-save it as UTF-8 CSV and upload again."
            ) from None
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"The CSV could not be parsed: {exc}".splitlines()[0]) from None
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        if name.endswith(".xls"):
            try:
                import xlrd  # noqa: F401
            except ImportError:
                raise IngestError("Legacy .xls files need the optional xlrd package. Save the workbook as .xlsx.") from None
        try:
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, dtype=object)
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"The worksheet could not be read: {type(exc).__name__}.") from None
        df = df.map(_cell_to_str)
    else:
        raise IngestError("Upload a .csv or .xlsx file.")
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) == 0:
        raise IngestError("No columns were found. Check that the first row contains headers.")
    if len(set(df.columns)) != len(df.columns):
        raise IngestError("Column headers must be unique.")
    if any(c.startswith("Unnamed:") for c in df.columns) and len(df) and _looks_headerless(df):
        raise IngestError("The first row does not look like a header row. Add column headers and upload again.")
    if len(df.columns) > MAX_COLUMNS:
        raise IngestError(f"The file has more than {MAX_COLUMNS} columns.")
    df = df.fillna("").astype(str).map(lambda v: v.strip())
    df = df.loc[~(df == "").all(axis=1)].reset_index(drop=True)
    if len(df) > MAX_ITEMS:
        raise IngestError(f"The file has {len(df)} rows. This proof of concept supports up to {MAX_ITEMS} items.")
    for col in df.columns:
        too_long = df[col].str.len() > MAX_CELL_CHARS
        if too_long.any():
            raise IngestError(f"Column '{col}' has a cell longer than {MAX_CELL_CHARS:,} characters (row {int(too_long.idxmax()) + 2}).")
    return df


def _cell_to_str(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _looks_headerless(df):
    unnamed = sum(c.startswith("Unnamed:") for c in df.columns)
    return unnamed >= max(1, len(df.columns) // 2)


# --------------------------------------------------------------------------- profiling

def infer_type(series):
    values = series[series != ""]
    if values.empty:
        return "empty"
    numeric = pd.to_numeric(values, errors="coerce")
    return "numeric" if numeric.notna().mean() >= 0.95 else "categorical" if values.nunique() <= max(20, len(values) // 5) else "text"


def profile_columns(df):
    rows = []
    for col in df.columns:
        s = df[col]
        nonempty = int((s != "").sum())
        rows.append({
            "column": col,
            "inferred_type": infer_type(s),
            "nonempty": nonempty,
            "null_pct": round(100 * (1 - nonempty / len(s)), 1) if len(s) else 0.0,
            "unique": int(s[s != ""].nunique()),
            "example": next((v[:80] for v in s if v), ""),
        })
    return pd.DataFrame(rows)


_ROLE_PATTERNS = {
    "item_id": r"^(item[_ ]?id|id|item|question[_ ]?id|qid)$",
    "key": r"^(key|answer|correct([_ ]?(answer|option|key))?|answer[_ ]?key)$",
    "domain": r"^(domain|content[_ ]?domain|area)$",
    "task": r"^(task|target[_ ]?task|competency)$",
    "category": r"^(category|topic|subdomain|blueprint)$",
    "difficulty": r"^(rasch[_ ]?b|b|difficulty|b[_ ]?param(eter)?|irt[_ ]?b)$",
    "p_value": r"^(p[_ ]?value|pvalue|p|prop[_ ]?correct)$",
    "point_biserial": r"^(point[_ ]?biserial|pbis|rpb|discrimination)$",
    "enemy": r"^(enemy|enemies|enemy[_ ]?ids?)$",
    "rationale": r"^(rationale|explanation)$",
    "family": r"^(family|stimulus[_ ]?id|case[_ ]?id|linked[_ ]?set|set[_ ]?id)$",
    "scale_group": r"^(scale|scale[_ ]?group|calibration|form|calibration[_ ]?group)$",
}
_TEXT_PATTERN = r"^(vignette|scenario|case|stem|question|lead[_ -]?in|prompt|item[_ ]?text)$"
_OPTION_PATTERN = r"^(choice|option|opt|alt|alternative|answer)[_ \-.]?([A-Za-z]|\d{1,2})$"


def detect_mapping(df):
    """Suggest a mapping; every suggestion remains editable."""
    cols = list(df.columns)
    lower = {c: c.strip().lower() for c in cols}
    mapping = ColumnMapping()
    used = set()
    for role, pattern in _ROLE_PATTERNS.items():
        match = next((c for c in cols if c not in used and re.match(pattern, lower[c])), None)
        if match:
            setattr(mapping, role, match)
            used.add(match)
    mapping.option_columns = [c for c in cols if re.match(_OPTION_PATTERN, lower[c])]
    used.update(mapping.option_columns)
    order = {"vignette": 0, "scenario": 0, "case": 0, "stem": 1, "question": 1, "item_text": 1, "item text": 1, "prompt": 1}
    text = [c for c in cols if c not in used and re.match(_TEXT_PATTERN, lower[c])]
    mapping.text_columns = sorted(text, key=lambda c: order.get(lower[c], 2))
    if not mapping.text_columns:
        longest = max(
            (c for c in cols if c not in used),
            key=lambda c: df[c].str.len().mean() if len(df) else 0,
            default=None,
        )
        mapping.text_columns = [longest] if longest else []
    return mapping


# --------------------------------------------------------------------------- records

def derive_option_ids(option_columns, override=None):
    if override:
        ids = [str(i).strip().upper() for i in override]
        if len(ids) != len(option_columns) or len(set(ids)) != len(ids) or not all(ids):
            raise IngestError("Option IDs must be unique and match the number of option columns.")
        return ids
    ids = []
    for pos, col in enumerate(option_columns):
        m = re.search(r"(?:^|[_ \-.])([A-Za-z]|\d{1,2})$", col.strip()) or re.fullmatch(r"([A-Za-z])", col.strip())
        token = m.group(1).upper() if m else None
        if token and token.isdigit():
            n = int(token)
            token = chr(ord("A") + n - 1) if 1 <= n <= 26 else None
        ids.append(token)
    if None in ids or len(set(ids)) != len(ids):
        ids = [chr(ord("A") + i) for i in range(len(option_columns))]
    return ids


def parse_key(raw, options):
    """Resolve an answer key without inventing one.

    Returns (set of option IDs, status) where status is single, multiple,
    missing, or unresolved.
    """
    raw = (raw or "").strip()
    if not raw:
        return set(), "missing"
    ids = {o["option_id"] for o in options}
    tokens = [t for t in re.split(r"[\s,;/|]+", raw) if t]
    upper = [t.upper().rstrip(".)") for t in tokens]
    if upper and all(t in ids for t in upper):
        found = set(upper)
    elif upper and all(t.isdigit() for t in upper):
        positions = [int(t) for t in upper]
        if not all(1 <= p <= len(options) for p in positions):
            return set(), "unresolved"
        found = {options[p - 1]["option_id"] for p in positions}
    else:
        matches = {o["option_id"] for o in options if o["text"].casefold() == raw.casefold()}
        if len(matches) != 1:
            return set(), "unresolved"
        found = matches
    return found, "single" if len(found) == 1 else "multiple"


def _normalize_id(value):
    value = (value or "").strip()
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    return value


def _column_types(df, mapping):
    types = {c: infer_type(df[c]) for c in df.columns}
    types.update({k: v for k, v in (mapping.type_overrides or {}).items() if k in types})
    for role in NUMERIC_ROLES:
        col = getattr(mapping, role)
        if col:
            types[col] = "numeric"
    return types


def validate_mapping(df, mapping):
    cols = set(df.columns)
    if not mapping.text_columns:
        raise IngestError("Select at least one text column to embed (for example the stem).")
    referenced = list(mapping.text_columns) + list(mapping.option_columns) + [
        getattr(mapping, r) for r in ("item_id", "key") + ROLE_FIELDS
    ]
    missing = sorted({c for c in referenced if c and c not in cols})
    if missing:
        raise IngestError(f"Mapped columns are not in the file: {', '.join(missing)}.")
    if mapping.item_id and mapping.item_id in mapping.text_columns:
        raise IngestError("The item ID column cannot also be embedded as item text.")
    overlap = set(mapping.text_columns) & set(mapping.option_columns)
    if overlap:
        raise IngestError(f"Columns cannot be both item text and an answer option: {', '.join(sorted(overlap))}.")
    if mapping.key and (mapping.key in mapping.text_columns or mapping.key in mapping.option_columns):
        raise IngestError("The answer key column must not be embedded or treated as an option.")
    for role in NUMERIC_ROLES:
        col = getattr(mapping, role)
        if not col:
            continue
        values = df[col]
        bad = values[(values != "") & pd.to_numeric(values, errors="coerce").isna()]
        if not bad.empty:
            rows = ", ".join(str(i + 2) for i in bad.index[:5])
            raise IngestError(
                f"Column '{col}' is mapped as {role.replace('_', ' ')} but has non-numeric values "
                f"(spreadsheet rows {rows}). Fix or clear those cells, or unmap the column."
            )
    if len({mapping.difficulty, mapping.p_value} - {None}) == 1 and mapping.difficulty and mapping.difficulty == mapping.p_value:
        raise IngestError("Map Rasch b and the classical p-value to different columns.")


def build_records(df, mapping):
    """Return (records, report). Records preserve every option and all metadata."""
    validate_mapping(df, mapping)
    types = _column_types(df, mapping)
    option_ids = derive_option_ids(mapping.option_columns, mapping.option_ids)
    labels = [mapping.label(c) for c in mapping.text_columns]
    if len(set(labels)) != len(labels):
        raise IngestError("Each embedded text column needs a distinct label.")
    report = {"rows_read": len(df), "exact_duplicates_removed": [], "skipped_empty_text": [],
              "synthetic_ids": False, "key_status": {}, "warnings": []}

    signatures = df.apply(lambda r: fingerprint(list(r.values)), axis=1)
    duplicated = signatures.duplicated(keep="first")
    for idx in df.index[duplicated]:
        first = int(signatures[signatures == signatures[idx]].index[0])
        report["exact_duplicates_removed"].append({"source_row": int(idx) + 2, "duplicate_of_row": first + 2})
    work = df.loc[~duplicated]

    if mapping.item_id:
        ids = work[mapping.item_id].map(_normalize_id)
        if (ids == "").any():
            rows = ", ".join(str(i + 2) for i in work.index[ids == ""][:5])
            raise IngestError(f"Item ID is blank on spreadsheet rows {rows}. Fill the IDs or unmap the ID column.")
        conflicts = sorted(ids[ids.duplicated(keep=False)].unique())
        if conflicts:
            raise IngestError(
                "These item IDs appear on rows with different content: " + ", ".join(conflicts[:10])
                + ". Give each item a unique ID or remove the conflicting rows."
            )
    else:
        ids = pd.Series([f"ROW-{i + 2:04d}" for i in work.index], index=work.index)
        report["synthetic_ids"] = True

    template = template_description(labels, mapping.include_options_in_embed)
    records = []
    for idx, row in work.iterrows():
        fields = {mapping.label(c): row[c] for c in mapping.text_columns}
        options = [
            {"option_id": oid, "text": row[col], "is_key": None, "source_column": col}
            for oid, col in zip(option_ids, mapping.option_columns) if row[col] != ""
        ]
        key_ids, key_status = parse_key(row[mapping.key] if mapping.key else "", options)
        if key_status in ("single", "multiple"):
            for o in options:
                o["is_key"] = o["option_id"] in key_ids
        report["key_status"][key_status] = report["key_status"].get(key_status, 0) + 1
        embed_text = build_embed_text(fields, labels, options, mapping.include_options_in_embed)
        if not embed_text.strip():
            report["skipped_empty_text"].append({"source_row": int(idx) + 2, "item_id": ids[idx]})
            continue
        metadata = {}
        for col in df.columns:
            value = row[col]
            if types.get(col) == "numeric" and value != "":
                value = float(value)
            metadata[col] = value if value != "" else None
        roles = {r: metadata.get(getattr(mapping, r)) if getattr(mapping, r) else None for r in ROLE_FIELDS}
        records.append({
            "internal_id": ids[idx],
            "item_id": ids[idx],
            "source_row": int(idx) + 2,
            "fields": fields,
            "field_order": labels,
            "options": options,
            "key_raw": row[mapping.key] if mapping.key else None,
            "key_status": key_status,
            "roles": roles,
            "metadata": metadata,
            "selected_embed_text": embed_text,
            "embed_text_hash": sha256_text(embed_text),
            "source_fingerprint": fingerprint({c: row[c] for c in df.columns}),
        })
    if not records:
        raise IngestError("No rows have text in the selected embedding columns.")
    report["items"] = len(records)
    report["template"] = template
    if mapping.include_options_in_embed:
        report["warnings"].append(
            "Option text is included in the embedded text. This increases lexical overlap between items "
            "and can surface answer-option leakage in similarity results."
        )
    for role in ("domain", "task", "category", "difficulty"):
        col = getattr(mapping, role)
        if col and col in mapping.text_columns:
            report["warnings"].append(f"The {role} column '{col}' is embedded; predictions on it would leak the label.")
    if mapping.p_value:
        vals = pd.to_numeric(df[mapping.p_value], errors="coerce").dropna()
        if ((vals < 0) | (vals > 1)).any():
            report["warnings"].append(f"Some p-values in '{mapping.p_value}' fall outside 0–1. Check the column mapping.")
    return records, report


def records_frame(records, columns=None):
    """A flat table of records for display/export (one row per unique item)."""
    rows = []
    for r in records:
        row = {"item_id": r["item_id"], "source_row": r["source_row"]}
        row.update({k: v for k, v in r["fields"].items()})
        for o in r["options"]:
            row[f"option_{o['option_id']}"] = o["text"]
        row["key"] = ",".join(o["option_id"] for o in r["options"] if o["is_key"]) or None
        row["key_status"] = r["key_status"]
        row.update({k: v for k, v in r["metadata"].items() if k not in row})
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[columns] if columns else frame
