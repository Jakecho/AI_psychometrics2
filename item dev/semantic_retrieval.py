"""Local search of explicitly selected bank records and clinical passages."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from math import isfinite, log
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from urllib.parse import urlsplit

import numpy as np
import yaml

from specialist_workflow import CLINICAL_CLASSES, task_for, validate_sources
from storage import hosted_mode


MAX_FILES = 256
MAX_RECORDS = 512
MAX_IMPORT_BYTES = 8 * 1024 * 1024
MAX_MARKDOWN_BYTES = 512 * 1024
MAX_PASSAGE_CHARS = 2400
PROVIDERS = {"openai", "anthropic"}
COLLECTIONS = {"bank", "clinical"}
TEXT_FIELDS = ("title", "citation", "url", "population", "setting", "decision", "decision_timing",
               "option_type", "reasoning_phase", "stage_title", "relative_file", "section")
BLUEPRINT_FIELDS = ("name", "version", "domain", "subdomain", "objective")
OBSERVED_FIELDS = {"item_id", "option_id", "item_version", "population", "administration", "sample_size",
                   "denominator", "numerator", "selected", "selection_count", "selection_rate",
                   "proportion", "frequency", "difficulty", "discrimination", "analysis_method",
                   "metric", "value", "source", "uncertainty"}

# Synonymous expressions share a dimension in addition to their individual words.
# These are vocabulary correspondences for discovery, never action or eligibility rules.
CONCEPTS = {
    "follow-up": ("follow up", "follow-up", "followup", "return visit", "subsequent visit", "reassessment"),
    "initial visit": ("initial presentation", "first encounter", "first visit", "initial evaluation", "initial assessment"),
    "report": ("lab result", "test result", "laboratory report", "final report", "result review"),
    "choose": ("choose", "select", "determine", "decide", "identify"),
    "investigation": ("investigation", "workup", "work-up", "diagnostic testing", "diagnostic evaluation"),
    "management": ("management", "treatment", "therapy", "intervention"),
    "referral": ("refer", "referral", "consultation", "specialist assessment"),
    "prevention": ("prevention", "preventive", "preventative", "prophylaxis"),
    "urgent": ("urgent", "emergent", "immediate", "without delay"),
    "setting": ("primary care", "family medicine", "general practice", "family practice"),
    "outpatient": ("outpatient", "ambulatory", "office based", "community clinic"),
    "inpatient": ("inpatient", "hospitalized", "hospitalised", "hospital ward"),
    "adult": ("adult", "adults", "grown-up", "grownups", "grown up"),
    "pediatric": ("pediatric", "paediatric", "child", "children", "childhood"),
    "older adult": ("older adult", "elderly", "geriatric", "senior citizen"),
    "pregnancy": ("pregnancy", "pregnant", "antenatal", "prenatal"),
    "asymptomatic": ("asymptomatic", "without symptoms", "no symptoms", "symptom free"),
    "annual": ("annually", "annual", "yearly", "one year", "12 months", "twelve months"),
    "repeat": ("repeat", "retest", "recollect", "recollection", "obtain another"),
    "sample": ("specimen", "sample"),
    "unusable": ("rejected", "unusable", "unsatisfactory", "inadequate specimen", "unreportable"),
    "FIT": ("fecal immunochemical testing", "fecal immunochemical test", "faecal immunochemical test",
            "immunochemical stool test", "stool immunoassay", "fit"),
    "stool": ("fecal", "faecal", "stool"),
    "colorectal": ("colorectal", "large bowel", "colon and rectal"),
    "colonoscopy": ("colonoscopy", "colonoscopic examination", "colonic endoscopy"),
    "hemoglobin": ("haemoglobin", "hemoglobin", "hgb"),
    "hypertension": ("high blood pressure", "elevated blood pressure", "hypertension", "htn"),
    "hypotension": ("low blood pressure", "hypotension"),
    "dyspnea": ("shortness of breath", "short of breath", "breathlessness", "dyspnea", "dyspnoea"),
    "syncope": ("loss of consciousness", "syncope", "fainting", "fainted"),
    "edema": ("edema", "oedema", "swelling"),
    "myocardial infarction": ("heart attack", "myocardial infarction"),
    "stroke": ("cerebrovascular accident", "stroke"),
    "diabetes": ("diabetes mellitus", "diabetes", "t2dm", "t1dm"),
    "hyperglycemia": ("high blood sugar", "elevated blood glucose", "hyperglycemia", "hyperglycaemia"),
    "renal": ("kidney", "renal"),
    "hepatic": ("liver", "hepatic"),
    "cardiac": ("cardiac", "heart"),
    "pulmonary": ("pulmonary", "lung"),
    "medication": ("medication", "medicine", "pharmacotherapy", "drug"),
    "adherence": ("adherence", "compliance", "taking as prescribed"),
    "adverse event": ("adverse event", "adverse effect", "side effect", "unwanted effect"),
    "contraindication": ("contraindication", "contraindicated", "reason to avoid"),
    "threshold": ("cut-off", "cutoff", "cut point", "threshold", "decision boundary"),
}
STOP_WORDS = frozenset("a an and are as at be by for from has in into is it of on or that the their then this to was what when which with".split())
WORDS = re.compile(r"[a-z]+[a-z0-9]*|\d+(?:\.\d+)?", re.I)
THRESHOLD = re.compile(r"(<=|>=|≤|≥|<|>)\s*([+−-]?\d+(?:\.\d+)?)")
_URL = re.compile(r"https?://[^\s<>\]\[()]+", re.I)


def _words(text):
    text = text.casefold()
    words = WORDS.findall(text)
    for operator, number in THRESHOLD.findall(text):
        words.append({"≤": "<=", "≥": ">="}.get(operator, operator) + number.replace("−", "-"))
    return words


_PHRASES = {}
for concept, alternatives in CONCEPTS.items():
    for phrase in alternatives:
        parts = tuple(_words(phrase))
        _PHRASES.setdefault(parts[0], []).append((parts, concept))


def _concepts(words):
    found = set()
    for index, token in enumerate(words):
        for parts, label in _PHRASES.get(token, ()):
            if tuple(words[index:index + len(parts)]) == parts:
                found.add(label)
    return found


def _terms(text):
    words = _words(text)
    tokens = Counter(w for w in words if w not in STOP_WORDS)
    for label in _concepts(words):
        tokens["concept:" + label] += 5
    return tokens


def _providers(values):
    if not isinstance(values, (list, tuple, set, frozenset)) or any(
        not isinstance(v, str) or v not in PROVIDERS for v in values
    ):
        raise ValueError("Provider permissions must list openai, anthropic, or neither.")
    return set(values)


def _relative_locator(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or PureWindowsPath(name).drive or any(part.startswith(".") for part in path.parts)
            or any(ord(c) < 32 for c in name)):
        raise ValueError("Use a relative, in-collection Markdown filename.")
    return path.as_posix()


def _text(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    return value.strip()


def _web_url(value):
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not (parsed.username or parsed.password):
            return value
    except ValueError:
        pass
    raise ValueError("Original source links must be HTTP or HTTPS citations.")


def _normalize_bank(raw, row):
    blueprint = raw.get("blueprint", {})
    if isinstance(blueprint, str):
        blueprint = {"domain": blueprint}
    if not isinstance(blueprint, dict):
        raise ValueError("Bank blueprint must be text or a blueprint object.")
    if blueprint:
        row["blueprint"] = {k: _text(blueprint[k], k) for k in BLUEPRINT_FIELDS if k in blueprint}
    if "options" in raw:
        if not isinstance(raw["options"], list):
            raise ValueError("Bank options must be a list.")
        options = []
        for entry in raw["options"]:
            if isinstance(entry, str):
                entry = {"text": entry}
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
                raise ValueError("Each bank option needs approved option text.")
            option = {"text": entry["text"]}
            if "option_id" in entry or "id" in entry:
                option["option_id"] = _text(entry.get("option_id", entry.get("id")), "option_id")
            options.append(option)
        row["options"] = options
    if "observed_option_evidence" in raw:
        evidence = raw["observed_option_evidence"]
        if not isinstance(evidence, list) or any(not isinstance(record, dict) for record in evidence):
            raise ValueError("Observed option evidence must be a list of records.")
        records = [{k: v for k, v in record.items() if k in OBSERVED_FIELDS} for record in evidence]
        if any(isinstance(value, (dict, list)) for record in records for value in record.values()):
            raise ValueError("Observed option evidence fields must be scalar values.")
        if any(isinstance(value, float) and not isfinite(value) for record in records for value in record.values()):
            raise ValueError("Observed option evidence must contain finite numbers.")
        row["observed_option_evidence"] = records


def _source(raw, collection, allowed):
    if not isinstance(raw, dict):
        raise ValueError("Each collection record must be an object.")
    source_class = raw.get("source_class", "bank" if collection == "bank" else "practice summary")
    valid_classes = {"bank"} if collection == "bank" else CLINICAL_CLASSES | {"practice summary"}
    if not isinstance(source_class, str) or source_class not in valid_classes:
        raise ValueError("A source class does not belong to the selected collection.")
    row = {"source_class": source_class, "providers": sorted(allowed & _providers(raw.get("providers", [])))}
    for field in ("source_id", "locator"):
        row[field] = _text(raw.get(field), field)
    row["source_date"] = _text(raw.get("source_date") or raw.get("version"), "source_date or version")
    row["excerpt"] = _text(raw.get("excerpt") or raw.get("summary"), "excerpt or approved summary")
    if len(row["excerpt"]) > 12000:
        raise ValueError("Supply an approved excerpt or summary of at most 12,000 characters.")
    for field in TEXT_FIELDS:
        if raw.get(field) is not None:
            row[field] = _text(raw[field], field)
    if row.get("relative_file"):
        row["relative_file"] = _relative_locator(row["relative_file"])
    citation = row.get("citation", "")
    if citation and (re.fullmatch(r"!?\[\[.+?\]\]", citation)
                     or citation.lower().startswith(("obsidian://", "file://"))
                     or re.fullmatch(r"[^\n]+\.md(?:#.*)?", citation, flags=re.I) and not _URL.search(citation)):
        row.pop("citation")
    if row.get("citation") and not row.get("url"):
        cited_url = _URL.search(row["citation"])
        if cited_url:
            row["url"] = cited_url.group().rstrip(".,;")
    if row.get("url"):
        row["url"] = _web_url(row["url"])
    if row.get("url") and not row.get("citation"):
        row["citation"] = row["url"]
    if collection == "clinical" and not row.get("citation"):
        row["source_class"] = "practice summary"
    for field in ("item_ids", "stage_ids"):
        if field in raw:
            if not isinstance(raw[field], list) or any(not isinstance(v, str) or not v for v in raw[field]):
                raise ValueError(f"{field} must be a list of stable IDs.")
            row[field] = list(raw[field])
    if collection == "bank":
        _normalize_bank(raw, row)
    return row


def _front_matter(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, lines
    end = next((i for i, line in enumerate(lines[1:256], 1) if line.strip() in {"---", "..."}), None)
    if end is None:
        raise ValueError("Close the Markdown properties block before the note text.")
    raw = "\n".join(lines[1:end])
    if len(raw) > 16384:
        raise ValueError("The Markdown properties block is too large.")
    try:
        # Inspect nodes without constructing arbitrary Python/YAML objects.
        node = yaml.compose(raw, Loader=yaml.BaseLoader)
    except (yaml.YAMLError, RecursionError):
        raise ValueError("A Markdown properties block could not be read.") from None
    metadata = {}
    seen = set()
    if node is not None and not isinstance(node, yaml.MappingNode):
        raise ValueError("Markdown properties must be named fields.")
    for key, value in (node.value if node is not None else ()):
        if not isinstance(key, yaml.ScalarNode) or key.value in seen or key.value == "<<":
            raise ValueError("Use unique, explicit names in Markdown properties.")
        seen.add(key.value)
        if key.value not in {*TEXT_FIELDS, "source", "reference", "source_class", "source_date", "date", "version", "providers",
                             "item_ids", "stage_ids", "approved", "retrieval_allowed"}:
            continue
        if key.value in ("source", "reference") and not isinstance(value, yaml.ScalarNode):
            continue
        if isinstance(value, yaml.ScalarNode):
            metadata[key.value] = value.value
        elif isinstance(value, yaml.SequenceNode) and all(isinstance(v, yaml.ScalarNode) for v in value.value):
            metadata[key.value] = [v.value for v in value.value]
        else:
            raise ValueError(f"Invalid Markdown property {key.value}.")
    return metadata, lines[end + 1:]


def _approved(properties):
    for k in ("approved", "retrieval_allowed"):
        if k in properties:
            value = properties[k]
            if isinstance(value, str):
                value = {"true": True, "false": False}.get(value.casefold())
            if type(value) is not bool:
                raise ValueError(f"{k} must be true or false.")
            if not value:
                return False
    return True


def _sections(lines):
    heading_path, body = [], []
    fenced = None
    for line in lines:
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if fenced is None:
                fenced = marker
            elif marker.startswith(fenced):
                fenced = None
            continue
        if fenced is not None:
            continue
        heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            if body:
                yield " / ".join(v for _, v in heading_path), "\n".join(body).strip()
                body = []
            level = len(heading.group(1))
            heading_path = [(n, title) for n, title in heading_path if n < level]
            heading_path.append((level, heading.group(2)))
        else:
            body.append(line)
    if body:
        yield " / ".join(v for _, v in heading_path), "\n".join(body).strip()


def _passages(section):
    """Break long sections at paragraph/word boundaries without splitting short notes."""
    remaining = section
    while remaining:
        end = len(remaining)
        if end > MAX_PASSAGE_CHARS:
            end = remaining.rfind("\n\n", MAX_PASSAGE_CHARS // 2, MAX_PASSAGE_CHARS)
            if end < 0:
                end = remaining.rfind(" ", MAX_PASSAGE_CHARS // 2, MAX_PASSAGE_CHARS)
            if end < 0:
                end = MAX_PASSAGE_CHARS
        yield remaining[:end]
        remaining = remaining[end:].lstrip()


def _markdown_records(content, name, allowed):
    if len(content) > MAX_MARKDOWN_BYTES:
        raise ValueError("A Markdown note exceeds the 512 KiB limit. Select a smaller note.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Use UTF-8 Markdown files for clinical reference passages.") from None
    metadata, lines = _front_matter(text)
    if not _approved(metadata):
        return []
    # Obsidian notes may record an original URL as a source/reference property.
    # A local vault path or internal note link is not an original publication.
    if not metadata.get("citation") and not metadata.get("url"):
        for alias in ("reference", "source"):
            reference = metadata.get(alias)
            if isinstance(reference, str) and _URL.search(reference):
                metadata["citation"] = reference
                break
    name = _relative_locator(name)
    version = metadata.get("source_date") or metadata.get("version") or metadata.get("date") or "sha256:" + sha256(content).hexdigest()[:16]
    props = metadata | {"relative_file": name, "title": metadata.get("title") or PurePosixPath(name).stem,
                        "source_date": version, "providers": metadata.get("providers", sorted(allowed))}
    occurrences = Counter()
    records = []
    for heading, section in _sections(lines):
        occurrences[heading] += 1
        locator = name + (f"#{heading}" if heading else "")
        if occurrences[heading] > 1:
            locator += f" (section {occurrences[heading]})"
        for i, excerpt in enumerate(_passages(section), 1):
            location = locator + (f" · passage {i}" if i > 1 else "")
            source_id = "md-" + sha256((location + "\0" + excerpt).encode()).hexdigest()[:24]
            raw = props | {"source_id": source_id, "locator": location, "section": heading, "excerpt": excerpt}
            # A URL in an unrelated paragraph is not evidence for every section.
            if not raw.get("citation") and not raw.get("url"):
                cited = re.search(r"(?im)^\s*(?:citation|reference|source):\s*(.+)$", excerpt)
                if cited:
                    raw["citation"] = cited.group(1).strip()
            records.append(_source(raw, "clinical", allowed))
            if len(records) > MAX_RECORDS:
                raise ValueError("Select a smaller collection with at most 512 passages.")
    return records


def import_collection(content, filename, collection, *, approved_providers=()):
    """Read a writer-selected upload in memory. File names are locators, never paths to open."""
    if collection not in COLLECTIONS:
        raise ValueError("Select the bank or clinical reference collection.")
    allowed = _providers(approved_providers)
    data = content.encode() if isinstance(content, str) else content
    if not isinstance(data, bytes) or len(data) > MAX_IMPORT_BYTES:
        raise ValueError("Supply a collection import of at most 8 MiB.")
    name = _relative_locator(_text(filename, "filename"))
    if collection == "clinical" and name.lower().endswith((".md", ".markdown")):
        rows = _markdown_records(data, name, allowed)
    elif name.lower().endswith(".json"):
        def unique(pairs):
            record = dict(pairs)
            if len(record) != len(pairs):
                raise ValueError("The collection contains duplicated JSON field names.")
            return record
        def invalid_number(_):
            raise ValueError("Collection data must contain finite JSON numbers.")
        try:
            records = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid_number)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise ValueError("The selected file must contain a UTF-8 JSON array of approved records.") from None
        if not isinstance(records, list) or len(records) > MAX_RECORDS or any(not isinstance(r, dict) for r in records):
            raise ValueError("Import up to 512 source records in a JSON array.")
        rows = [_source(raw, collection, allowed) for raw in records if _approved(raw)]
    else:
        raise ValueError("Use a JSON collection file or upload a clinical Markdown note.")
    validate_sources(rows)
    return rows


def load_markdown_folder(folder, *, approved_providers=()):
    """Read visible regular Markdown files under this one directory, without following links."""
    if hosted_mode():
        raise ValueError("Use an approved clinical import in the hosted workspace.")
    if not isinstance(folder, (str, Path)) or not str(folder).strip():
        raise ValueError("Select a local Markdown directory first.")
    allowed = _providers(approved_providers)
    root = Path(folder).expanduser()
    if not root.is_absolute():
        raise ValueError("Select an absolute Markdown folder path.")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    records, totals = [], Counter()

    def visit(descriptor, prefix):
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = entry.name
                if name.startswith(".") or entry.is_symlink():
                    continue
                relative = f"{prefix}/{name}" if prefix else name
                totals["entries"] += 1
                if totals["entries"] > 8192 or relative.count("/") > 32:
                    raise ValueError("Select a smaller subfolder of approved Markdown notes.")
                if entry.is_dir(follow_symlinks=False):
                    child = os.open(name, flags | os.O_DIRECTORY, dir_fd=descriptor)
                    try:
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif name.lower().endswith((".md", ".markdown")) and entry.is_file(follow_symlinks=False):
                    totals["files"] += 1
                    if totals["files"] > MAX_FILES:
                        raise ValueError("Select a folder containing at most 256 Markdown files.")
                    fd = os.open(name, flags | os.O_NONBLOCK, dir_fd=descriptor)
                    with os.fdopen(fd, "rb") as handle:
                        st = os.fstat(handle.fileno())
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        content = handle.read(min(MAX_MARKDOWN_BYTES, MAX_IMPORT_BYTES - totals["bytes"]) + 1)
                        totals["bytes"] += len(content)
                        if totals["bytes"] > MAX_IMPORT_BYTES:
                            raise ValueError("Select a folder containing at most 8 MiB of Markdown.")
                    records.extend(_markdown_records(content, relative, allowed))
                    if len(records) > MAX_RECORDS:
                        raise ValueError("Select a smaller collection with at most 512 passages.")
    try:
        fd = os.open(root, flags | os.O_DIRECTORY)
        try:
            visit(fd, "")
        finally:
            os.close(fd)
    except OSError:
        raise ValueError("The selected directory or one of its notes could not be read. Select a readable Markdown folder.") from None
    validate_sources(records)
    return sorted(records, key=lambda r: (r["relative_file"], r["locator"], r["source_id"]))


def build_query(case, item, exact_terms=""):
    """Describe the task and current stage without including patient text, options, or keys."""
    stage = next((s for s in case["stages"] if s["id"] == item["stage_id"]), None)
    if stage is None:
        raise ValueError("Choose an item with an existing disclosure stage.")
    task = task_for(case, item)
    action_types = dict.fromkeys(o["option_type"] for o in item["options"] if o.get("option_type"))
    timing = item.get("decision_workflow", {}).get("decision_timing") or stage.get("elapsed") or stage["title"]
    return {
        "item_id": item["id"], "stage_id": stage["id"], "stage_title": stage["title"],
        "objective": task["objective"], "blueprint": " ".join(str(task["blueprint"].get(k) or "")
                                                           for k in ("domain", "subdomain")).strip(),
        "decision": task["decision"], "assessment_intent": task["assessment_intent"],
        "reasoning_phase": task.get("appraisal_dimension") or task["reasoning_phase"],
        "population": task["population"], "setting": task["setting"], "decision_timing": timing,
        "option_type": " ".join(action_types), "exact_terms": _text(exact_terms, "Exact terms"),
    }


def _document_text(source):
    bp = source.get("blueprint", {})
    fields = [source.get(k, "") for k in ("title", "section", "excerpt", "decision", "decision_timing",
                                           "stage_title", "reasoning_phase", "option_type", "population", "setting")]
    if isinstance(bp, dict):
        fields.extend(bp.get(k, "") for k in ("domain", "subdomain", "objective"))
    fields.extend(o["text"] for o in source.get("options", []))
    return " ".join(fields)


def _similarities(documents, query):
    """TF-IDF concept vectors and a low-rank semantic space learned from this authorized corpus."""
    rows = [_terms(doc) for doc in documents]
    query_terms = _terms(query)
    df = Counter(word for row in rows for word in row.keys())
    # Keep query terms first so a large vocabulary cannot crowd out named tests or thresholds.
    terms = sorted(df, key=lambda w: (w not in query_terms, -df[w], w))[:8192]
    columns = {word: i for i, word in enumerate(terms)}
    idf = {word: 1 + log((1 + len(rows)) / (1 + df[word])) for word in terms}
    vectors = np.zeros((len(rows) + 1, len(terms)), dtype=np.float64)
    for vector, frequencies in zip(vectors, rows + [query_terms]):
        for term, frequency in frequencies.items():
            if term in columns:
                vector[columns[term]] = (1 + log(frequency)) * idf[term]
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
    matrix, query_vector = vectors[:-1], vectors[-1]
    cosine = matrix @ query_vector
    if len(rows) < 4 or len(terms) < 4 or not np.any(query_vector):
        return cosine
    # Randomized truncated SVD keeps the work bounded for an imported collection.
    # Fit only the permission-filtered documents; the query is projected afterwards.
    rank = min(16, max(2, int(np.sqrt(len(rows)))), len(terms) - 1)
    sample = np.random.default_rng(0).standard_normal((len(terms), min(rank + 4, len(rows))))
    basis, _ = np.linalg.qr(matrix @ sample, mode="reduced")
    _, singular_values, components = np.linalg.svd(basis.T @ matrix, full_matrices=False)
    projection = components[:rank][singular_values[:rank] > 1e-8]
    latent = matrix @ projection.T
    q = projection @ query_vector
    denom = np.linalg.norm(latent, axis=1) * np.linalg.norm(q)
    contextual = np.divide(latent @ q, denom, out=np.zeros_like(cosine), where=denom > 1e-9)
    return np.clip(0.75 * cosine + 0.25 * np.maximum(contextual, 0), 0, 1)


def _scope_match(source, query):
    score = 0.0
    reasons = []
    for field, weight in (("population", .12), ("decision_timing", .10), ("stage_title", .08),
                          ("setting", .07), ("decision", .08), ("option_type", .06), ("reasoning_phase", .04)):
        value = source.get(field)
        wanted = query.get(field)
        if not value or not wanted:
            continue
        record_words = _terms(value)
        target_words = _terms(wanted)
        shared = record_words.keys() & target_words.keys()
        total = max(1, min(len(target_words), len(record_words)))
        agreement = len(shared) / total
        if set(_words(value)) == set(_words(wanted)) or agreement >= .5:
            score += weight
            reasons.append("Matches " + field.replace("_", " ") + ": " + value)
        elif any(k.startswith("concept:") for k in shared):
            score += weight * .5
            reasons.append("Related " + field.replace("_", " ") + ": " + value)
        elif field in {"population", "decision_timing", "stage_title", "setting", "option_type"}:
            score -= weight
            reasons.append("Different " + field.replace("_", " ") + ": " + value)
    return score, reasons


def search(records, query, collection, *, provider=None, limit=12):
    """Rank records inside one authorized collection. No data is transmitted."""
    if collection not in COLLECTIONS or (provider is not None and provider not in PROVIDERS):
        raise ValueError("Choose a supported collection and provider permission.")
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise ValueError("Search at most 512 records per collection.")
    if type(limit) is not int or limit < 1:
        raise ValueError("Choose a positive result count.")
    candidates = []
    valid_classes = {"bank"} if collection == "bank" else CLINICAL_CLASSES | {"practice summary"}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Collection entries must be source records.")
        source_class = row.get("source_class")
        if not _approved(row) or not isinstance(source_class, str) or source_class not in valid_classes:
            continue
        if provider and provider not in _providers(row.get("providers", [])):
            continue
        if row.get("item_ids") and query["item_id"] not in row["item_ids"]:
            continue
        if row.get("stage_ids") and query["stage_id"] not in row["stage_ids"]:
            continue
        candidates.append(row)
    if not candidates:
        return []
    validate_sources(candidates)
    query_fields = ("objective", "decision", "assessment_intent", "blueprint", "reasoning_phase",
                    "population", "setting", "decision_timing", "stage_title", "option_type", "exact_terms")
    query_text = " ".join(query.get(k, "") for k in query_fields)
    documents = [_document_text(source) for source in candidates]
    similarities = _similarities(documents, query_text)
    exact = set(_words(query.get("exact_terms", ""))) - STOP_WORDS
    concepts = _concepts(_words(query_text))
    results = []
    for source, document, similarity in zip(candidates, documents, similarities):
        words = _words(document)
        exact_matches = exact.intersection(words)
        scope_score, reasons = _scope_match(source, query)
        if similarity < 1e-6 and not exact_matches:
            continue
        common = sorted(concepts & _concepts(words))
        if common:
            reasons.append("Shared concepts: " + ", ".join(common[:6]))
        if exact_matches:
            reasons.append("Exact search terms: " + ", ".join(sorted(exact_matches)))
        if not reasons:
            reasons.append("Related wording or term context in the authorized collection.")
        exact_score = len(exact_matches) / len(exact) if exact else 0.0
        score = .72 * float(similarity) + scope_score + .12 * exact_score
        results.append({"source": deepcopy(source), "similarity": round(float(np.clip(similarity, 0, 1)), 6),
                        "score": round(score, 6), "reasons": reasons})
    return sorted(results, key=lambda r: (-r["score"], -r["similarity"], r["source"]["source_id"]))[:limit]
